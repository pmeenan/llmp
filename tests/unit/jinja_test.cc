// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The Jinja-subset interpreter (chat/jinja.h): every snippet fixture against
// transformers' rendering (docs/experiments/chat-template-corpus), the
// provenance of rendered text, and every bound against hostile templates.

#include "chat/jinja.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/json.h"
#include "expected_error.h"
#include "tokenizer_fixtures.h"

namespace {

namespace jinja = jitllm::chat::jinja;
namespace json = jitllm::base::json;
using jitllm::test_support::Failed;
using jitllm::test_support::Get;
using jitllm::test_support::LoadJson;

using Variables = std::vector<std::pair<std::string, jinja::Input>>;

const jinja::CivilTime kNow{.year = 2026,
                            .month = 10,
                            .day = 2,
                            .hour = 12,
                            .minute = 34,
                            .second = 56,
                            .weekday = 5,
                            .yearday = 274};

std::expected<jinja::Rendered, jinja::Error> Render(std::string_view source,
                                                    const Variables& variables = {},
                                                    const jinja::Limits& limits = {}) {
  auto t = jinja::Template::Parse(source, limits);
  if (!t) {
    return std::unexpected(t.error());
  }
  return t->Render(variables, kNow);
}

std::string Text(std::string_view source, const Variables& variables = {}) {
  auto r = Render(source, variables);
  if (!r) {
    return "ERROR " + r.error().ToString();
  }
  return r->text;
}

void AddObject(json::Value object, Variables& variables) {
  for (std::size_t i = 0; i < object.size(); ++i) {
    variables.emplace_back(std::string(object.key(i)), jinja::Input::Json(object.member(i)));
  }
}

TEST(JinjaSnippets, MatchTransformers) {
  const json::Document doc = LoadJson("chat/jinja-snippets.json");
  const json::Value snippets = Get(doc.root(), "snippets");
  ASSERT_GT(snippets.size(), 100U);
  std::size_t texts = 0;
  std::size_t errors = 0;
  for (std::size_t i = 0; i < snippets.size(); ++i) {
    const json::Value s = snippets.at(i);
    const std::string_view name = Get(s, "name").string();
    Variables variables;
    AddObject(Get(doc.root(), "variables"), variables);
    if (const auto extra = s.find("variables")) {
      AddObject(*extra, variables);
    }
    const auto rendered = Render(Get(s, "template").string(), variables);
    if (const auto j = s.find("jitllm"); j && j->string() == "unsupported") {
      EXPECT_EQ(Failed(rendered, &jinja::Error::code), jinja::Code::kUnsupported) << name;
      continue;
    }
    if (const auto text = s.find("text")) {
      ASSERT_TRUE(rendered.has_value()) << name << ": " << rendered.error().ToString();
      EXPECT_EQ(rendered->text, text->string()) << name;
      ++texts;
    } else {
      EXPECT_FALSE(rendered.has_value())
          << name << " rendered, but transformers raised " << Get(s, "error").string();
      ++errors;
    }
  }
  EXPECT_GE(texts, 80U);
  EXPECT_GE(errors, 15U);
}

TEST(Jinja, RaisedAndRuntimeErrorsAreTold) {
  EXPECT_EQ(Failed(Render("{{ raise_exception('x') }}"), &jinja::Error::code),
            jinja::Code::kRaised);
  EXPECT_EQ(Failed(Render("{{ x.y }}"), &jinja::Error::code), jinja::Code::kRuntime);
  EXPECT_EQ(Failed(Render("{% if %}"), &jinja::Error::code), jinja::Code::kSyntax);
  EXPECT_EQ(Failed(Render("{{ x | nofilter }}"), &jinja::Error::code), jinja::Code::kUnsupported);
  const auto e = Render("a\nb\n{{ x.y }}");
  ASSERT_FALSE(e.has_value());
  EXPECT_EQ(e.error().line, 3U) << e.error().ToString();
}

TEST(Jinja, FromJsonParsesAsLlamaCppDoes) {
  EXPECT_EQ(Text("{% set v = '{\"a\": [1, 2.50, null, true, \"\\u00e9\"], \"b\": 1E2}'|from_json %}"
                 "{{ v.a }} {{ v.b }} {{ v|tojson }}"),
            "[1, 2.5, None, True, 'é'] 100.0 {\"a\": [1, 2.5, null, true, \"é\"], \"b\": 100.0}");
  EXPECT_EQ(Failed(Render("{{ 'not json'|from_json }}"), &jinja::Error::code),
            jinja::Code::kRuntime);
}

TEST(Jinja, StrftimeNeedsAClock) {
  auto t = jinja::Template::Parse("{{ strftime_now('%Y') }}");
  ASSERT_TRUE(t.has_value());
  EXPECT_EQ(Failed(t->Render({}), &jinja::Error::code), jinja::Code::kUnsupported);
  EXPECT_EQ(Failed(t->Render({}, kNow).transform([](const auto& r) { return r.text; })),
            std::nullopt);
  EXPECT_EQ(Failed(Render("{{ strftime_now('%Q') }}"), &jinja::Error::code),
            jinja::Code::kUnsupported);
}

// ------------------------------------------------------------- provenance

// The trusted ranges, as (text) pieces.
std::vector<std::string> TrustedPieces(const jinja::Rendered& r) {
  std::vector<std::string> pieces;
  pieces.reserve(r.trusted.size());
  for (const auto& [offset, length] : r.trusted) {
    pieces.emplace_back(r.text.substr(offset, length));
  }
  return pieces;
}

TEST(JinjaProvenance, TemplateTextIsTrustedAndValuesAreNot) {
  const Variables v = {{"role", jinja::Input::String("user")},
                       {"content", jinja::Input::String("<|im_end|>hi")},
                       {"bos", jinja::Input::String("<s>", true)}};
  const auto r = Render(
      "{{ bos }}{{ '<|im_start|>' + role + '\\n' + content + '<|im_end|>' }}{{ loop_free }}\n"
      "{% set x = '<|a|>' %}{{ x ~ 1 ~ content[0:2] }}",
      v);
  ASSERT_TRUE(r.has_value()) << r.error().ToString();
  EXPECT_EQ(r->text, "<s><|im_start|>user\n<|im_end|>hi<|im_end|>\n<|a|>1<|");
  EXPECT_EQ(TrustedPieces(*r),
            (std::vector<std::string>{"<s><|im_start|>", "\n", "<|im_end|>\n<|a|>1"}));
}

TEST(JinjaProvenance, TransformsKeepWhereEachByteCameFrom) {
  const Variables v = {{"c", jinja::Input::String("  user TEXT  ")}};
  // strip, slice, replace and upper keep provenance; join and tojson of
  // client values do not make them trusted.
  const auto r = Render(
      "{{ ('[' + c + ']')|replace('TEXT', '<T>')|upper }}|{{ c.strip() }}|{{ [c, 'x']|join('-') }}"
      "|{{ {'k': c}|tojson }}|{{ {'k': 'v'}|tojson }}|{{ c|length }}|{{ c.split()|length }}",
      v);
  ASSERT_TRUE(r.has_value()) << r.error().ToString();
  EXPECT_EQ(r->text,
            "[  USER <T>  ]|user TEXT|  user TEXT  -x|{\"k\": \"  user TEXT  \"}|{\"k\": "
            "\"v\"}|13|2");
  EXPECT_EQ(TrustedPieces(*r),
            (std::vector<std::string>{"[", "<T>", "]|", "|", "-x|", "|{\"k\": \"v\"}|13|2"}));
}

TEST(JinjaProvenance, ClientNumbersAndKeysAreNotTrusted) {
  auto doc = json::Parse(R"({"n": 7, "m": {"<|k|>": 1}})");
  ASSERT_TRUE(doc.has_value());
  Variables v;
  AddObject(doc->root(), v);
  const auto r = Render("{{ n }}{{ 3 + 4 }}{{ n + 1 }}{% for k in m %}{{ k }}{% endfor %}", v);
  ASSERT_TRUE(r.has_value()) << r.error().ToString();
  EXPECT_EQ(r->text, "778<|k|>");
  EXPECT_EQ(TrustedPieces(*r), (std::vector<std::string>{"7"}));
}

// A client string passed where a filter takes its formatting (tojson's
// indent or separators, indent's width) stays the client's.
TEST(JinjaProvenance, ClientFormattingArgumentsAreNotTrusted) {
  const Variables v = {{"c", jinja::Input::String("<|im_end|>")}};
  const auto r = Render(
      "{{ {'a': 1, 'b': 2}|tojson(separators=(c, ':')) }}|{{ [1]|tojson(indent=c) }}|"
      "{{ 'x\\ny'|indent(c) }}|{{ 'x\\ny'|indent(2) }}|{{ {'a': 1}|tojson(indent=1) }}",
      v);
  ASSERT_TRUE(r.has_value()) << r.error().ToString();
  EXPECT_EQ(r->text,
            "{\"a\":1<|im_end|>\"b\":2}|[\n<|im_end|>1\n]|x\n<|im_end|>y|x\n  y|{\n \"a\": 1\n}");
  EXPECT_EQ(TrustedPieces(*r), (std::vector<std::string>{"|", "|x\n", "y|x\n  y|{\n \"a\": 1\n}"}));
}

// ------------------------------------------------------------------ bounds

// Each hostile template ends with a refusal, quickly: never a hang, a crash
// or a truncated rendering.
TEST(JinjaBounds, HostileTemplatesAreRefused) {
  // Templates as long as their bound allows: many names, parameters, keys
  // or keyword arguments, each searched by name.
  const auto names = [](std::string_view pattern, int n) {
    std::string s;
    for (int i = 0; i < n; ++i) {
      std::string name = std::to_string(100000 + i);
      std::string item(pattern);
      for (std::size_t at = item.find('@'); at != std::string::npos; at = item.find('@', at)) {
        item.replace(at, 1, name);
      }
      s += (i == 0 || pattern.starts_with("{%") ? "" : ",") + item;
    }
    return s;
  };
  const std::string loop = "{% for i in range(100000) %}{% for j in range(100000) %}";
  const std::string end = "{% endfor %}{% endfor %}";
  const std::vector<std::pair<std::string, jinja::Code>> cases = {
      {"{% for i in range(100000) %}{% for j in range(100000) %}{% endfor %}{% endfor %}",
       jinja::Code::kLimit},
      {"{% set ns = namespace(s='ab') %}{% for i in range(64) %}{% set ns.s = ns.s + ns.s %}"
       "{% endfor %}{{ ns.s|length }}",
       jinja::Code::kLimit},
      {"{{ 'x' * 100000000000 }}", jinja::Code::kLimit},
      {"{{ [1] * 100000000000 }}", jinja::Code::kLimit},
      {"{% macro f(n) %}{{ f(n + 1) }}{% endmacro %}{{ f(0) }}", jinja::Code::kLimit},
      {"{% macro f() %}{{ f() ~ f() }}{% endmacro %}{{ f() }}", jinja::Code::kLimit},
      {"{{ range(100001)|length }}", jinja::Code::kLimit},
      {"{% set ns = namespace(l=[]) %}{% for i in range(100) %}{% set ns.l = [ns.l] %}"
       "{% endfor %}{{ ns.l|tojson }}",
       jinja::Code::kLimit},
      {"{% for i in range(100000) %}{{ 'abcdefghij' * 1000 }}{% endfor %}", jinja::Code::kLimit},
      {"{% set ns = namespace(l=[]) %}{% for i in range(100000) %}{% set ns.l = ns.l + "
       "['abcdefghijabcdefghij' * 100] %}{% endfor %}",
       jinja::Code::kLimit},
      {"{{ 2 ** 100 }}", jinja::Code::kUnsupported},
      {"{{ 9223372036854775807 + 1 }}", jinja::Code::kUnsupported},
      // One string held many times over: its text, JSON, format or indent
      // is bounded as it grows, not after.
      {"{% set b = 'x' * 1048576 %}{% set l = [b] * 100000 %}{{ l|tojson }}", jinja::Code::kLimit},
      {"{% set b = 'x' * 1048576 %}{% set l = [b] * 100000 %}{{ l }}", jinja::Code::kLimit},
      {"{% set b = 'x' * 1048576 %}{% set a = [b, b, b, b, b, b, b, b] %}"
       "{% set a = [a, a, a, a, a, a, a, a] %}{% set a = [a, a, a, a, a, a, a, a] %}"
       "{{ a|string|length }}",
       jinja::Code::kLimit},
      {"{% set b = 'x' * 1048576 %}{{ (('%s' * 100000) % ((b,) * 100000))|length }}",
       jinja::Code::kLimit},
      {"{{ (('a\\n' * 1000000)|indent('x' * 1000000))|length }}", jinja::Code::kLimit},
      {"{% set ns = namespace(l=[1]) %}{% for i in range(60) %}{% set ns.l = [ns.l, ns.l] %}"
       "{% endfor %}{{ (ns.l|tojson(indent='x' * 1000000))|length }}",
       jinja::Code::kLimit},
      {"{{ [1]|tojson(indent=100000000000) }}", jinja::Code::kLimit},
      // Comparisons and searches cost what they scan.
      {"{% set b = 'x' * 1048576 %}{% set l = [b] * 1000000 %}{{ l == l }}", jinja::Code::kLimit},
      {"{% set b = 'x' * 1048576 %}{% set l = [b] * 1000000 %}{{ l|unique|length }}",
       jinja::Code::kLimit},
      {"{% set h = 'a' * 4000000 %}{% set n = ('a' * 2000000) ~ 'b' %}"
       "{% for i in range(1000) %}{% set x = n in h %}{% endfor %}",
       jinja::Code::kLimit},
      {"{% set h = 'a' * 4000000 %}{% set n = ('a' * 2000000) ~ 'b' %}"
       "{% for i in range(1000) %}{% set x = h.split(n) ~ h.rsplit(n) ~ h.count(n) %}{% endfor %}",
       jinja::Code::kLimit},
      {"{% set s = 'x' * 60000000 %}{% set c = 'abcdefghijklmnopqrstuvw' * 1000 %}"
       "{% for i in range(1000) %}{% set y = s.strip(c) %}{% endfor %}",
       jinja::Code::kLimit},
      // strip's character set: its sort and per-text-code-point membership are
      // charged, so a huge set over strippable text meets the work bound.
      {"{% set c = 'abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRST' * 900000 %}"
       "{% set s = 'a' * 60000000 %}{{ s.strip(c)|length }}",
       jinja::Code::kLimit},
      {"{% set c = 'abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRST' * 900000 %}"
       "{% set s = 'a' * 30000000 %}{% for i in range(30) %}{% set y = s.strip(c) %}{% endfor %}",
       jinja::Code::kLimit},
      {"{{ ('b' * 20000000) in 'x' }}", jinja::Code::kLimit},  // a search's tables
      // Searches by name cost what they compare: a scope's variables, a
      // macro's parameters, a mapping literal's keys, namespace() and a
      // filter's keyword arguments.
      {names("{%set a@=1%}", 20000) + loop + "{{ z999999 }}" + end, jinja::Code::kLimit},
      {"{% for i in range(100000) %}" + names("{%set a@=1%}", 20000) + "{% endfor %}",
       jinja::Code::kLimit},
      {"{% macro m(" + names("a@=0", 5000) + ") %}{% endmacro %}" + loop + "{{ m() }}" + end,
       jinja::Code::kLimit},
      {"{% macro m(" + names("a@", 5000) + ") %}{% endmacro %}" + loop + "{{ m(" +
           names("a@=0", 5000) + ") }}" + end,
       jinja::Code::kLimit},
      {loop + "{% set d = {" + names("'k@':1", 5000) + "} %}" + end, jinja::Code::kLimit},
      {loop + "{% set d = namespace(" + names("k@=1", 5000) + ") %}" + end, jinja::Code::kLimit},
      {"{% set l = range(1000)|list %}" + loop + "{% set x = l|map('default', " +
           names("defaul@=1", 5000) + ")|list %}" + end,
       jinja::Code::kLimit},
      {"{% set " + std::string(400000, 'q') + "a = 1 %}" + loop + "{{ " + std::string(400000, 'q') +
           "b }}" + end,
       jinja::Code::kLimit},
      // A mapping or namespace searched by a huge key: hashed or compared.
      {"{% set b = 'x' * 60000000 %}{% set d = dict(" + names("k@=1", 20) + ") %}" + loop +
           "{% if b in d %}{% endif %}{% set x = d.get(b) %}" + end,
       jinja::Code::kLimit},
      {"{% set b = ('x' * 30000000) ~ 'a' %}{% set c = ('x' * 30000000) ~ 'b' %}"
       "{% set d = {b: 1} %}{% set n = namespace(d) %}" +
           loop + "{% set x = c in d %}{% set y = n[c] %}" + end,
       jinja::Code::kLimit},
      // Prefix and suffix tests compare up to their length.
      {"{% set b = ('x' * 30000000) ~ 'a' %}{% set c = ('x' * 30000000) ~ 'b' %}" + loop +
           "{% set x = b.startswith(c) ~ b.endswith(c) ~ b.removeprefix(c) %}" + end,
       jinja::Code::kLimit},
      {"{% set b = 'x' * 2000 %}{% set t = (('x' * 999) ~ 'y',) * 1000000 %}" + loop +
           "{% set x = b.startswith(t) %}" + end,
       jinja::Code::kLimit},
      // Scans that build nothing: case tests, parses, formats.
      {"{% set b = 'x' * 60000000 %}" + loop + "{% set x = b is lower %}" + end,
       jinja::Code::kLimit},
      {"{% set b = (' ' * 60000000) ~ '0' %}" + loop + "{% set x = b|from_json %}" + end,
       jinja::Code::kLimit},
      {"{% set b = '1' * 60000000 %}" + loop + "{% set x = b|int ~ b|float %}" + end,
       jinja::Code::kLimit},
      {"{% set f = '%-d' * 20000000 %}" + loop + "{% set x = strftime_now(f) %}" + end,
       jinja::Code::kLimit},
      {"{% set f = '%s' * 2000000 %}{% set a = ('',) * 2000000 %}" + loop + "{% set x = f % a %}" +
           end,
       jinja::Code::kLimit},
      // tojson sorts a mapping's keys each time it prints it.
      {"{% set d = dict(" + names("k@=1", 20000) + ") %}" + loop +
           "{% set x = d|tojson(sort_keys=true) %}" + end,
       jinja::Code::kLimit},
  };
  for (const auto& [source, code] : cases) {
    const auto started = std::chrono::steady_clock::now();
    const auto r = Render(source);
    const auto seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    EXPECT_EQ(Failed(r, &jinja::Error::code), code) << source;
    EXPECT_LT(seconds, 30.0) << source;
  }
}

// A loop's own copy of what it iterates (a filtered list) is held while it
// runs: loops nested over one long list meet the live bound, not memory's.
TEST(JinjaBounds, LoopCopiesAreHeld) {
  const std::string source =
      "{% set l = (range(100000)|list) * 20 %}"
      "{% for a in l if true %}{% for b in l if true %}{% for c in l if true %}"
      "{% endfor %}{% endfor %}{% endfor %}";
  EXPECT_EQ(Failed(Render(source), &jinja::Error::reason),
            "the template's values held more bytes than their bound");
}

// Chains no depth bound covers (each loop iteration wraps the last link)
// are released without recursion: no stack overflow when they die.
TEST(JinjaBounds, LongChainsAreReleasedWithoutRecursion) {
  EXPECT_EQ(Text("{% set ns = namespace(x={}) %}{% for i in range(100000) %}{% for j in range(5) %}"
                 "{% set ns.x = {'a': ns.x.copy} %}{% endfor %}{% endfor %}done"),
            "done");
  EXPECT_EQ(Text("{% set h = namespace(cur=namespace(x=none)) %}{% set root = h.cur %}"
                 "{% for i in range(100000) %}{% for j in range(5) %}{% set c = h.cur %}"
                 "{% set c.x = namespace(x=none) %}{% set h.cur = c.x %}{% endfor %}{% endfor %}"
                 "{% set h.cur = none %}done"),
            "done");
}

// A bound reached where no later step reports it (a comparison's work, the
// last thing the template does) still refuses the rendering: Equal stops
// early with a meaningless answer, which must never be printed.
TEST(JinjaBounds, ABoundReachedLastIsStillRefused) {
  const std::string big(1000, 'x');
  const Variables v = {{"a", jinja::Input::String(big)}, {"b", jinja::Input::String(big)}};
  jinja::Limits limits;
  limits.max_work_bytes = 2500;  // the two inputs, and not their comparison
  EXPECT_EQ(Failed(Render("{{ a == b }}", v, limits), &jinja::Error::code), jinja::Code::kLimit);
  limits.max_work_bytes = 4000;
  EXPECT_EQ(Text("{{ a == b }}", v), "True");
  EXPECT_EQ(Render("{{ a == b }}", v, limits).transform([](const auto& r) { return r.text; }),
            "True");
}

TEST(JinjaBounds, DeepNestingIsRefusedWhenParsed) {
  const std::size_t n = 10000;
  EXPECT_EQ(Failed(Render("{{ " + std::string(n, '(') + "1" + std::string(n, ')') + " }}"),
                   &jinja::Error::code),
            jinja::Code::kLimit);
  EXPECT_EQ(Failed(Render("{{ " + std::string(n, '[') + std::string(n, ']') + " }}"),
                   &jinja::Error::code),
            jinja::Code::kLimit);
  std::string ifs;
  for (std::size_t i = 0; i < n; ++i) {
    ifs += "{% if true %}";
  }
  EXPECT_EQ(Failed(Render(ifs), &jinja::Error::code), jinja::Code::kLimit);
  std::string nots = "{{ ";
  for (std::size_t i = 0; i < n; ++i) {
    nots += "not ";
  }
  EXPECT_EQ(Failed(Render(nots + "1 }}"), &jinja::Error::code), jinja::Code::kLimit);
  std::string minus = "{{ ";
  for (std::size_t i = 0; i < n; ++i) {
    minus += "-";
  }
  EXPECT_EQ(Failed(Render(minus + "1 }}"), &jinja::Error::code), jinja::Code::kLimit);
  std::string filters = "{{ 1";
  for (std::size_t i = 0; i < n * 10; ++i) {
    filters += "|abs";
  }
  EXPECT_EQ(Failed(Render(filters + " }}"), &jinja::Error::code), jinja::Code::kLimit);
}

TEST(JinjaBounds, LimitsAreConfigurable) {
  jinja::Limits small;
  small.max_template_bytes = 16;
  EXPECT_EQ(Failed(Render("{{ 'a long template' }}", {}, small), &jinja::Error::code),
            jinja::Code::kLimit);
  jinja::Limits output;
  output.max_output_bytes = 1000;
  EXPECT_EQ(
      Failed(Render("{% for i in range(1000) %}xx{% endfor %}", {}, output), &jinja::Error::code),
      jinja::Code::kLimit);
  EXPECT_TRUE(Render("{% for i in range(400) %}xx{% endfor %}", {}, output).has_value());
  jinja::Limits steps;
  steps.max_steps = 1000;
  EXPECT_EQ(Failed(Render("{% for i in range(1000) %}{{ i }}{% endfor %}", {}, steps),
                   &jinja::Error::code),
            jinja::Code::kLimit);
}

// Large client values stay fast: lookups in a big mapping are indexed.
TEST(JinjaBounds, LargeClientMappingsStayFast) {
  std::string big = "{";
  for (int i = 0; i < 100000; ++i) {
    big +=
        (i == 0 ? "" : ",") + std::string("\"k") + std::to_string(i) + "\": " + std::to_string(i);
  }
  big += "}";
  auto doc = json::Parse(big);
  ASSERT_TRUE(doc.has_value());
  const Variables v = {{"m", jinja::Input::Json(doc->root())}};
  const auto started = std::chrono::steady_clock::now();
  const auto r = Render(
      "{% set ns = namespace(t=0) %}{% for k in m %}{% set ns.t = ns.t + m[k] %}{% endfor %}"
      "{{ ns.t }}",
      v);
  ASSERT_TRUE(r.has_value()) << r.error().ToString();
  EXPECT_EQ(r->text, "4999950000");
  EXPECT_LT(std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(),
            20.0);
}

// Random byte soup and truncations of a real template never crash.
TEST(JinjaFuzz, MutatedTemplatesFailCleanly) {
  const std::string base =
      "{%- set ns = namespace(n=0) -%}{% for m in messages if m.role != 'x' %}"
      "{{- '<|' ~ m.role ~ '|>' + (m.content|trim) -}}{% if loop.last %}{{ ns.n }}{% endif %}"
      "{% set ns.n = ns.n + 1 %}{% endfor %}{{ raise_exception('done') if false else 'ok' }}";
  auto doc = json::Parse(R"({"messages": [{"role": "user", "content": " hi "}]})");
  ASSERT_TRUE(doc.has_value());
  Variables v;
  AddObject(doc->root(), v);
  EXPECT_EQ(Text(base, v), "<|user|>hi0ok");
  std::uint32_t seed = 12345;
  const auto next = [&]() {
    seed = (seed * 1103515245U) + 12345U;
    return seed >> 8U;
  };
  constexpr std::string_view kAlphabet = "{}%#-+|.,:()[]'\"~ \nabcdefgimnorstx01";
  for (int i = 0; i < 20000; ++i) {
    std::string t = base;
    const std::uint32_t edits = 1 + (next() % 4);
    for (std::uint32_t k = 0; k < edits && !t.empty(); ++k) {
      const std::size_t at = next() % t.size();
      switch (next() % 3) {
        case 0:
          t[at] = kAlphabet[next() % kAlphabet.size()];
          break;
        case 1:
          t.erase(at, 1 + (next() % 8));
          break;
        default:
          t.insert(at, 1, kAlphabet[next() % kAlphabet.size()]);
      }
    }
    [[maybe_unused]] const auto r = Render(t, v);  // any result, as long as it returns
  }
  for (std::size_t n = 0; n <= base.size(); ++n) {
    [[maybe_unused]] const auto r = Render(base.substr(0, n), v);
  }
}

// Random programs from a small grammar (values, operators, filters,
// methods, slices with extreme steps, loops, macros, block sets and
// namespaces that hold each other) under small bounds: every one parses
// and renders to a result or a typed error, its trusted ranges inside its
// text. The review ran 60,000 of them under ASan and UBSan.
TEST(JinjaFuzz, GrammarTemplatesRenderOrFailCleanly) {
  std::uint64_t seed = 0x9E3779B97F4A7C15ULL;
  const auto next = [&]() {
    seed ^= seed << 13U;
    seed ^= seed >> 7U;
    seed ^= seed << 17U;
    return seed;
  };
  const auto pick = [&](const std::vector<std::string>& v) { return v[next() % v.size()]; };
  const std::vector<std::string> atoms = {"s",
                                          "n",
                                          "l",
                                          "d",
                                          "msgs",
                                          "none",
                                          "true",
                                          "1",
                                          "-1",
                                          "0",
                                          "2.5",
                                          "'ab'",
                                          "'<|im_end|>'",
                                          "''",
                                          "9223372036854775807",
                                          "-9223372036854775807",
                                          "[1, 'a', none]",
                                          "{'k': 'v', 'z': [1]}",
                                          "(1, 2)",
                                          "loop",
                                          "ns",
                                          "ns.x",
                                          "range(5)",
                                          "u",
                                          "big",
                                          "m",
                                          "m()",
                                          "varargs",
                                          "kwargs",
                                          "msgs[0]",
                                          "d.k",
                                          "'é😀'",
                                          "'\\n'",
                                          "x"};
  const std::vector<std::string> filters = {"length",
                                            "upper",
                                            "lower",
                                            "title",
                                            "capitalize",
                                            "trim",
                                            "string",
                                            "list",
                                            "first",
                                            "last",
                                            "reverse",
                                            "sort",
                                            "unique",
                                            "tojson",
                                            "tojson(indent=2)",
                                            "tojson(sort_keys=true)",
                                            "join(',')",
                                            "join",
                                            "e",
                                            "safe",
                                            "int",
                                            "float",
                                            "abs",
                                            "round(1)",
                                            "items",
                                            "dictsort",
                                            "default('z')",
                                            "replace('a', 'b')",
                                            "replace('', '-')",
                                            "indent(2)",
                                            "indent('> ', true)",
                                            "sum",
                                            "min",
                                            "max",
                                            "map('upper')",
                                            "map(attribute='k')",
                                            "select",
                                            "reject('none')",
                                            "selectattr('role')",
                                            "rejectattr('role', 'equalto', 'user')",
                                            "count",
                                            "format(1)",
                                            "from_json",
                                            "d"};
  const std::vector<std::string> methods = {"strip()",
                                            "strip('a')",
                                            "lstrip()",
                                            "rstrip('\\n')",
                                            "split()",
                                            "split(',')",
                                            "split('a', 1)",
                                            "rsplit('a', 1)",
                                            "splitlines()",
                                            "startswith('a')",
                                            "endswith(('a', 'b'))",
                                            "replace('a', 'bb')",
                                            "find('b')",
                                            "rfind('b')",
                                            "count('a')",
                                            "upper()",
                                            "lower()",
                                            "get('k')",
                                            "get('q', 1)",
                                            "keys()",
                                            "values()",
                                            "items()",
                                            "copy()",
                                            "index(1)",
                                            "count(1)",
                                            "removeprefix('a')",
                                            "removesuffix('b')",
                                            "join(['x', 'y'])",
                                            "append(1)",
                                            "pop()"};
  const std::vector<std::string> operators = {
      " + ",  " - ",  " * ", " / ",  " // ", " % ",      " ** ",  " ~ ",
      " == ", " != ", " < ", " >= ", " in ", " not in ", " and ", " or "};
  const std::vector<std::string> tests = {
      "defined",  "undefined", "none",         "string", "number", "mapping",
      "iterable", "sequence",  "callable",     "odd",    "even",   "divisibleby(3)",
      "eq(1)",    "in([1])",   "sameas(none)", "lower",  "true"};
  const std::vector<std::string> steps = {
      "1", "-1", "2", "-9223372036854775807", "9223372036854775807", "none"};
  std::function<std::string(int)> expr = [&](int depth) -> std::string {
    if (depth <= 0) {
      return pick(atoms);
    }
    switch (next() % 10) {
      case 0:
        return "(" + expr(depth - 1) + pick(operators) + expr(depth - 1) + ")";
      case 1:
        return expr(depth - 1) + "|" + pick(filters);
      case 2:
        return "(" + expr(depth - 1) + ")." + pick(methods);
      case 3:
        return "(" + expr(depth - 1) + ")[" + expr(0) + "]";
      case 4:
        return "(" + expr(depth - 1) + ")[" + pick(atoms) + ":" + pick(atoms) + ":" + pick(steps) +
               "]";
      case 5:
        return "(" + expr(depth - 1) + " is " + (next() % 2 != 0 ? "not " : "") + pick(tests) + ")";
      case 6:
        return "(" + expr(depth - 1) + " if " + expr(depth - 1) + " else " + expr(depth - 1) + ")";
      case 7:
        return "[" + expr(depth - 1) + ", " + expr(depth - 1) + "]";
      case 8:
        return "{'a': " + expr(depth - 1) + ", " + expr(depth - 1) + ": 1}";
      default:
        return "(not " + expr(depth - 1) + ")";
    }
  };
  std::function<std::string(int)> stmt = [&](int depth) -> std::string {
    if (depth <= 0) {
      return "{{ " + expr(2) + " }}";
    }
    switch (next() % 9) {
      case 0:
        return "{% for x in " + expr(1) + (next() % 3 == 0 ? " if " + expr(1) : "") + " %}" +
               stmt(depth - 1) + "{{ loop.index }}{{ loop.previtem }}{% if " + expr(1) +
               " %}{% break %}{% endif %}{% else %}E{% endfor %}";
      case 1:
        return "{% if " + expr(2) + " %}" + stmt(depth - 1) + "{% elif " + expr(1) + " %}" +
               stmt(depth - 1) + "{% else %}" + stmt(depth - 1) + "{% endif %}";
      case 2:
        return "{% set x = " + expr(2) + " %}" + stmt(depth - 1);
      case 3:
        return "{% set ns.x = " + expr(2) + " %}" + stmt(depth - 1);
      case 4:
        return "{% macro m(a, b=" + expr(1) + ") %}" + stmt(depth - 1) +
               "{{ a }}{{ varargs }}{% endmacro %}" + stmt(depth - 1);
      case 5:
        return "{% set x %}" + stmt(depth - 1) + "{% endset %}{{ x|length }}";
      case 6:
        return "{%- if " + expr(1) + " -%}\n  " + stmt(depth - 1) + "\n{%+ endif %}";
      case 7:
        return "{% set ns.y = namespace(z=ns) %}{% set ns.x = [ns.y, ns] %}" + stmt(depth - 1);
      default:
        return stmt(depth - 1) + stmt(depth - 1);
    }
  };
  auto doc = json::Parse(
      R"({"s": "  Héllo, a,b <x> 😀 ", "n": 42, "l": [3, 1, 2, "a", null],
          "d": {"k": "v", "items": 1, "b": [1, {"c": 2}]},
          "msgs": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": null}],
          "u": "é", "big": 123456789012345678901234567890})");
  ASSERT_TRUE(doc.has_value());
  Variables v;
  AddObject(doc->root(), v);
  jinja::Limits limits;
  limits.max_steps = 200'000;
  limits.max_work_bytes = std::uint64_t{16} << 20U;
  limits.max_live_bytes = std::size_t{16} << 20U;
  limits.max_string_bytes = std::size_t{1} << 20U;
  limits.max_output_bytes = std::size_t{1} << 20U;
  std::size_t rendered = 0;
  for (int i = 0; i < 3000; ++i) {
    const std::string t = "{% set ns = namespace(x=1) %}{% macro m() %}M{% endmacro %}" + stmt(3);
    auto program = jinja::Template::Parse(t, limits);
    ASSERT_TRUE(program.has_value()) << t << ": " << program.error().ToString();
    const auto r = program->Render(v, kNow);
    if (r) {
      ++rendered;
      for (const auto& [offset, length] : r->trusted) {
        ASSERT_LE(offset + length, r->text.size()) << t;
      }
    }
  }
  EXPECT_GT(rendered, 100U);
}

}  // namespace
