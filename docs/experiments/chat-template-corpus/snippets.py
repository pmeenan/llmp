#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
"""Renders the interpreter's snippet fixtures (README.md).

    snippets.py --out tests/unit/data/chat/jinja-snippets.json

Each snippet is a small template written for llmpalooza's tests, rendered by
transformers' compiled chat-template environment (the one apply_chat_template
uses) with the snippet's JSON variables and strftime_now fixed at
2026-10-02 12:34:56. The fixture records the text or the exception;
`llmp` names what llmpalooza does differently on purpose ("unsupported": the
subset refuses it).
"""

import sys

sys.dont_write_bytecode = True

import argparse
import datetime
import json

CTX = {
    "msgs": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"},
             {"role": "assistant", "content": "Hello!", "reasoning_content": "greet"}],
    "d": {"b": 1, "A": 2, "a": 3, "items": "key-items"},
    "s": "  Héllo, Wörld! 東京 😀  ",
    "n": 42, "f": 2.5, "neg": -7, "t": True, "nothing": None,
    "nested": {"x": {"y": [1, {"z": "deep"}]}},
    "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Weather <now> & \"later\"",
                                                "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                               "required": ["city"]}}}],
    "big": 123456789012345678901234567890, "fl": [1.0, 1e16, 1e-05, 0.1, -0.0, 3.14159],
    "uni": "é  \u0085\x01'\"\\",
}

S = []


def add(name, template, llmp=None, **ctx):
    S.append({"name": name, "template": template, "llmp": llmp, "extra": ctx})


# Output, literals and whitespace control.
add("print-constants", "{{ none }}|{{ None }}|{{ true }}|{{ False }}|{{ 1 }}|{{ 1.0 }}|{{ -0.0 }}|{{ 1e20 }}|{{ 1_000 }}")
add("print-containers", "{{ [1,'a',none,true,1.0,{'k':'v'}] }}|{{ ('a',) }}|{{ () }}|{{ {'a':\"it's\"} }}|{{ msgs[0] }}")
add("print-unicode-repr", "{{ [uni] }}|{{ uni }}")
add("string-escapes", "{{ 'a\\nb\\tc\\u00e9\\x41\\101\\U0001F600\\q\\\\' }}|{{ \"q'\" }}|{{ 'a' 'b' \"c\" }}")
add("trailing-newline", "{{ 'a' }}\n")
add("trailing-newlines", "{{ 'a' }}\n\n")
add("crlf", "a\r\nb\rc{% if true %}\r\n{{ 'x' }}{% endif %}\r\n")
add("lstrip-trim", "  {% if true %}\n  x\n  {% endif %}\n  {# c #}\n  y\n")
add("lstrip-not-variable", "  {{ 'v' }}\n  {% if true %}t{% endif %}  \n")
add("lstrip-plus", "  {%+ if true %}a{% endif %}\n")
add("trim-plus", "{% if true +%}\na{% endif %}")
add("minus-control", "{%- if true -%}  a  {%- endif -%}  b {{- ' c ' -}} d\n {#- x -#} e")
add("minus-unicode-space", "a　{%- if true %}b{% endif %}")
add("comment-trim", "a\n{# comment #}\nb")
add("raw-block", "{% raw %}{{ not evaluated }}{% endraw %}|{%- raw -%}  x  {%- endraw -%}|")
add("text-only", "plain text, no tags\nsecond line")
add("lstrip-after-text-no-newline", "x  {% if true %}y{% endif %}")
add("block-at-start-indented", "    {% set a = 1 %}{{ a }}")
add("multiline-tag", "{%\n  set a = [\n 1,\n 2,\n ]\n%}{{ a }}")
# Arithmetic and operators.
add("arithmetic", "{{ 3/2 }} {{ 4/2 }} {{ 7//2 }} {{ -7//2 }} {{ 7%3 }} {{ -7%3 }} {{ 2**10 }} {{ 0.1+0.2 }} {{ true + 1 }} {{ 7.5//2 }} {{ -7.5%2 }} {{ 2**-1 }}")
add("precedence", "{{ - 2 ** 2 }} {{ 2 ** 3 ** 2 }} {{ 1 + 2 * 3 }} {{ (1 + 2) * 3 }} {{ -1|abs }} {{ not 1 == 2 }} {{ 1 ~ (2 + 3) }} {{ 1 + 2 ~ 3 }}")
add("precedence-concat-plus", "{{ 1 ~ 2 + 3 }}")
add("logic", "{{ 1 and 'x' }} {{ 0 or 'y' }} {{ none or [] }} {{ 'a' and none }} {{ not '' }} {{ 1 < 2 < 3 }} {{ 3 > 2 > 2 }}")
add("comparisons", "{{ 1 == 1.0 }} {{ true == 1 }} {{ 1 in [true] }} {{ 'a' in 'cat' }} {{ 'k' in {'k':1} }} {{ x in [] }} {{ 'a' < 'b' }} {{ [1,2] < [1,3] }} {{ (1,2) == [1,2] }} {{ none == none }} {{ x == y }} {{ 'a' not in 'b' }}")
add("concat", "{{ [1,2] + [3] }} {{ 'ab' * 3 }} {{ 3 * 'ab' }} {{ [0] * 2 }} {{ 'a' ~ 1 ~ none ~ true ~ x }} {{ 'x' * 0 }} {{ 'x' * -1 }}")
add("ternary", "{{ 'x' if false }}|{{ ('x' if false) is defined }}|{{ 'a' if true else 'b' }}|{{ 'a' if false else 'b' if true else 'c' }}")
add("string-format-op", "{{ '%s and %d' % ('a', 3) }}|{{ '%s' % 'x' }}|{{ '100%%' % () }}")
add("format-filter", "{{ '%s and %s'|format('a', 1) }} {{ '%d'|format(3) }} {{ '%s'|format({'a': 'b'}) }}")
# Names, attributes and items.
add("attr-item", "{{ msgs[1].content }} {{ msgs[1]['content'] }} {{ nested.x.y[1].z }} {{ nested['x']['y'][0] }} {{ msgs[-1].role }} {{ msgs[9] is defined }}")
add("attr-missing", "{{ msgs[0].nope is defined }} {{ nothing.attr is defined }}")
add("undefined-chain", "{{ x.y.z }}")
add("undefined-item", "{{ x[0] }}")
add("undefined-call", "{{ x() }}")
add("dict-method-shadow", "{{ d['items'] }} {{ d.items is defined }} {{ d.get('a') }} {{ d.get('zz') }} {{ d.get('zz', 9) }}")
add("dict-methods", "{{ d.keys()|list }} {{ d.values()|list }} {{ d.items()|list }}")
add("slices", "{{ 'abc'[::-1] }} {{ 'héllo'[1] }} {{ 'héllo'[1:3] }} {{ [1,2,3][-1] }} {{ [1,2,3][:-1] }} {{ 'abc'[10:] }} {{ [1,2,3,4,5][::2] }} {{ [1,2,3,4,5][4:1:-1] }} {{ 'abc'[-10:2] }} {{ (1,2,3)[1:] }} {{ s[2:7] }}")
add("index-strings", "{{ 'héllo'|length }} {{ s|length }} {{ s[2] }} {{ 'abc'[-1] }} {{ 'abc'[5] is defined }}")
# Statements.
add("loop-scope", "{% for x in [1,2,3] %}{% if loop.first %}{% set a = 'A' %}{% endif %}[{{ a }}]{% endfor %}|{{ a }}")
add("if-scope", "{% if true %}{% set b = 2 %}{% endif %}{{ b }}")
add("macro-late-binding", "{% macro m() %}{{ v }}{% endmacro %}{% set v = 1 %}{{ m() }}{% set v = 2 %}{{ m() }}")
add("macro-in-loop", "{% for x in [1] %}{% set v = 5 %}{% macro m2() %}{{ v }}{% endmacro %}{{ m2() }}{% endfor %}")
add("macro-args", "{% macro f(a, b=2) %}{{ a }}-{{ b }}{% endmacro %}{{ f(1) }} {{ f(1, b=3) }} {{ f(b=4, a=0) }} {{ f() }}")
add("macro-varargs", "{% macro f(a) %}{{ a }}{{ varargs }}{{ kwargs }}{% endmacro %}{{ f(1, 2, 3, k='v') }}")
add("macro-too-many", "{% macro f(a) %}{{ a }}{% endmacro %}{{ f(1, 2) }}")
add("macro-default-uses-earlier", "{% macro f(a, b=a~'!') %}{{ b }}{% endmacro %}{{ f('x') }}")
add("macro-recursion", "{% macro r(n) %}{{ n }}{% if n > 0 %}{{ r(n - 1) }}{% endif %}{% endmacro %}{{ r(5) }}")
add("macro-returns-string", "{% macro m() %} x {% endmacro %}[{{ m()|trim }}]{{ m() is string }}")
add("namespace", "{% set ns = namespace(a=1, b='x') %}{% for i in range(3) %}{% set ns.a = ns.a + i %}{% endfor %}{{ ns.a }} {{ ns['a'] }} {{ ns.b }} {{ ns.c is defined }}")
add("namespace-from-dict", "{% set ns = namespace({'a': 1}, b=2) %}{{ ns.a }}{{ ns.b }}")
add("set-on-non-namespace", "{% set d.x = 1 %}")
add("set-tuple", "{% set a, b = 1, 2 %}{{ a }}{{ b }}{% set (c, e) = ['x', 'y'] %}{{ c }}{{ e }}")
add("set-block", "{% set x %}  inner {{ 1 }} {% endset %}[{{ x }}]{% set y | trim | upper %} yy {% endset %}[{{ y }}]")
add("for-items", "{% for k, v in {'a':1,'b':2}|items %}{{ k }}={{ v }};{% endfor %}{% for a in [] %}x{% else %}empty{% endfor %}")
add("for-filter", "{% for x in [1,2,3] if x > 1 %}{{ loop.index }}/{{ loop.length }}{{ loop.previtem }}{% endfor %}")
add("for-loop-vars", "{% for x in 'abc' %}{{ loop.index0 }}{{ loop.revindex }}{{ loop.revindex0 }}{{ loop.first }}{{ loop.last }}{{ loop.nextitem }}{{ loop.cycle('o','e') }}{{ loop.depth }};{% endfor %}")
add("for-dict", "{% for k in d %}{{ k }},{% endfor %}")
add("for-nested-loop", "{% for a in [1,2] %}{% for b in [3,4] %}{{ loop.index }}{{ a }}{{ b }} {% endfor %}{{ loop.index }}|{% endfor %}")
add("for-break-continue", "{% for i in range(5) %}{% if i == 1 %}{% continue %}{% endif %}{% if i == 3 %}{% break %}{% endif %}{{ i }}{% endfor %}")
add("for-unpack", "{% for a, b in [[1,2],[3,4]] %}{{ a + b }}{% endfor %}")
add("for-unpack-wrong", "{% for a, b in [[1,2,3]] %}{{ a }}{% endfor %}")
add("for-over-none", "{% for a in nothing %}{{ a }}{% endfor %}")
add("for-over-int", "{% for a in n %}{{ a }}{% endfor %}")
add("loop-outside", "{{ loop }}|{{ loop is defined }}")
add("elif-chain", "{% for v in [0, 1, 2, 3] %}{% if v == 0 %}zero{% elif v == 1 %}one{% elif v == 2 %}two{% else %}many{% endif %} {% endfor %}")
add("generation-tag", "{%- generation -%} gen {%- endgeneration -%}|{% generation %}x{% endgeneration %}")
add("raise", "{{ raise_exception('boom') }}")
add("raise-in-branch", "{% if msgs|length > 2 %}{{ raise_exception('too many') }}{% endif %}ok")
add("strftime", "{{ strftime_now('%d %b %Y') }}|{{ strftime_now('%Y-%m-%d %H:%M:%S %A %a %B %j %p %I %y') }}|{{ strftime_now('%-d %-m') }}")
add("range", "{{ range(3)|list }} {{ range(1,7,2)|list }} {{ range(5)|length }} {{ range(5, 0, -2)|list }} {{ range(0)|list }}")
add("range-too-big", "{{ range(1000000)|length }}")
add("dict-global", "{{ dict(a=1, b='x') }}")
# Filters.
add("filter-strings", "{{ ' x '|trim }}|{{ 'xxhixx'|trim('x') }}|{{ 'Hello World'|upper }}|{{ 'hello world'|title }}|{{ 'hELLO'|capitalize }}|{{ 'ABC'|lower }}|{{ s|trim }}")
add("filter-default", "{{ x|default('d') }} {{ ''|default('d') }} {{ ''|default('d', true) }} {{ none|default('d') }} {{ x|d('dd') }} {{ x|default }}|")
add("filter-numbers", "{{ 5|string + 'x' }} {{ 2.0|int }} {{ '3'|int + 1 }} {{ 'x'|int }} {{ '2.5'|float }} {{ 2.567|round(1) }} {{ -3|abs }} {{ ' 12 '|int }} {{ '1_0'|int }} {{ '2.9'|int }} {{ 2.5|round }} {{ 3.5|round }} {{ 2.4|round(0, 'ceil') }} {{ 'x'|float }} {{ n|float }}")
add("filter-first-last", "{{ [1,2,3]|first }} {{ [1,2,3]|last }} {{ 'abc'|first }} {{ []|first is defined }} {{ [1,2]|reverse|list }} {{ 'abc'|reverse }} {{ 'abc'|last }}")
add("filter-select", "{{ [{'a':1},{'a':2},{'b':3}]|selectattr('a','defined')|list }} {{ [{'r':'s'},{'r':'u'}]|rejectattr('r','equalto','s')|list }} {{ [{'n':'x'}]|map(attribute='n')|join(',') }} {{ [1,2,3,4]|select('odd')|list }} {{ [1,2,3,4]|reject('even')|list }} {{ [0,1,'',none]|select|list }} {{ msgs|selectattr('role', 'eq', 'user')|map(attribute='content')|first }}")
add("filter-map", "{{ ['a','B']|map('upper')|list }} {{ [1,2]|map('string')|list }} {{ msgs|map(attribute='role')|list }} {{ [{'a':1},{}]|map(attribute='a', default=0)|list }} {{ nested.x.y|map(attribute='z', default='-')|list }}")
add("filter-sort", "{{ [3,1,2]|sort }} {{ {'b':1,'A':2,'a':3}|dictsort }} {{ ['b','A','a']|sort }} {{ ['b','A','a']|sort(case_sensitive=true) }} {{ [3,1]|sort(reverse=true) }} {{ msgs|sort(attribute='role')|map(attribute='role')|list }} {{ {'b':1,'a':2}|dictsort(by='value') }} {{ {'b':1,'a':2}|dictsort(reverse=true) }}")
add("filter-tojson", "{{ {'a':[1,2.5,'é',none]}|tojson }} {{ {'a':1}|tojson(indent=2) }} {{ 'é'|tojson(ensure_ascii=True) }} {{ {'a':1,'b':[]}|tojson(separators=(',',':')) }} {{ {'b':1,'a':2}|tojson(sort_keys=true) }} {{ '😀'|tojson(true) }}")
add("filter-tojson-values", "{{ [1,[2,[3]]]|tojson }} {{ none|tojson }} {{ true|tojson }} {{ 'a\"b\\\\\\n\\x01\\x7f'|tojson }} {{ 1.0|tojson }} {{ fl|tojson }} {{ big|tojson }} {{ tools|tojson }} {{ tools[0]|tojson(indent=4) }}")
add("filter-tojson-empty", "{{ {}|tojson(indent=2) }}|{{ []|tojson(indent=2) }}|{{ {'a':{}}|tojson(indent=1) }}|{{ [[]]|tojson(indent='\\t') }}|{{ {'a': [1]}|tojson(indent=0) }}")
add("filter-tojson-undefined", "{{ x|tojson }}")
add("filter-length", "{{ [1,2,3]|length }} {{ {'a':1}|length }} {{ x|length }} {{ 'é'|count }} {{ ''|length }}")
add("filter-length-none", "{{ nothing|length }}")
add("filter-join", "{{ ['a','b']|join }} {{ [1,2]|join('-') }} {{ ['a', none]|join(',') }} {{ msgs|join(', ', attribute='role') }}")
add("filter-items", "{{ d|items|list }} {{ x|items|list }}")
add("filter-items-list", "{{ [1]|items|list }}")
add("filter-unique-min-max-sum", "{{ [1,2,1]|unique|list }} {{ ['a','A','b']|unique|list }} {{ [1,5,3]|max }} {{ [1,5,3]|min }} {{ [1,2]|sum }} {{ [{'v':2},{'v':3}]|sum(attribute='v') }} {{ []|max is defined }} {{ ['b','A']|min }}")
add("filter-list", "{{ 'ab'|list }} {{ d|list }} {{ x|list }} {{ (1,2)|list }}")
add("filter-replace", "{{ 'aaa'|replace('a','b') }} {{ 'aaa'|replace('a','b', 2) }} {{ 'abc'|replace('','-') }} {{ ('x'|safe)|replace('x','<') }} {{ s|replace(' ', '') }}")
add("filter-indent", "{{ 'x'|indent(2) }}|{{ 'a\\nb\\n\\nc'|indent(2) }}|{{ 'a\\nb'|indent(2, true) }}|{{ 'a\\n\\nb'|indent(2, blank=true) }}|{{ 'a\\nb'|indent('> ') }}")
add("filter-safe-escape", "{{ 'a<' + ('<b>'|safe) }}|{{ ('<b>'|safe) + '<' }}|{{ ('<b>'|safe) ~ '<' }}|{{ ['<',('<'|safe)]|join }}|{{ '<&>\"\\''|e }}|{{ ('<'|safe) is escaped }}|{{ ('x'|safe)|trim is escaped }}|{{ (('a'|safe) + '&') ~ 'b' }}")
add("filter-string", "{{ none|string }} {{ [1]|string }} {{ 1.5|string }} {{ (1, 'a')|string }}")
add("filter-unknown", "{{ 'x'|wordcount }}", llmp="unsupported")
add("test-unknown", "{{ 'x' is truthy }}", llmp="unsupported")
# Tests.
add("tests-types", "{{ 'abc' is string }} {{ 1 is number }} {{ true is number }} {{ true is integer }} {{ {} is mapping }} {{ 'a' is sequence }} {{ {} is sequence }} {{ 'a' is iterable }} {{ 1 is iterable }} {{ none is none }} {{ x is undefined }} {{ 1 is boolean }} {{ false is false }} {{ 1.0 is float }} {{ 1 is float }} {{ big is integer }} {{ x is iterable }} {{ x is sequence }}")
add("tests-values", "{{ 2 is even }} {{ 3 is odd }} {{ 9 is divisibleby 3 }} {{ 9 is divisibleby(2) }} {{ 1 is eq 1 }} {{ 'a' is in 'abc' }} {{ 1 is sameas 1 }} {{ 'a' is lower }} {{ 'A' is upper }} {{ 2 is gt 1 }} {{ 2 is lessthan 1 }} {{ 1 is ne 2 }} {{ 1 is ge 1 }} {{ 1 is le 0 }} {{ x is not defined }} {{ msgs is not string }} {{ (x is defined) and x }}")
add("tests-callable", "{% macro m() %}{% endmacro %}{{ m is callable }} {{ 'a'.upper is callable }} {{ 1 is callable }}")
add("test-none-arg", "{{ x is none if true else 'n' }}")
# Python string methods.
add("methods", "{{ 'A-b'.lower() }} {{ 'ab'.startswith(('x','a')) }} {{ 'abc'.replace('b','B') }} {{ ' x'.lstrip() }} {{ 'a.b.c'.split('.', 1) }} {{ 'abc'.endswith('c') }} {{ 'a b '.rstrip() }} {{ 'xxaxx'.strip('x') }} {{ 'abc'.upper() }} {{ 'a,b'.rsplit(',', 1) }} {{ 'ab'.find('b') }} {{ 'ab'.rfind('z') }} {{ 'aXbXc'.count('X') }} {{ '-'.join(['a','b']) }} {{ 'hello world'.title() }} {{ 'abc'.capitalize() }}")
add("methods-split", "{{ 'a,b,,c'.split(',') }} {{ ' a  b '.split() }} {{ ' a b c '.split(none, 1) }} {{ ' a b c '.rsplit(none, 1) }} {{ 'a\\nb\\r\\nc'.splitlines() }} {{ s.split() }} {{ ''.split() }} {{ ''.split(',') }}")
add("methods-split-empty-sep", "{{ 'abc'.split('') }}")
add("methods-prefix", "{{ 'prefix-x'.removeprefix('prefix-') }} {{ 'x.txt'.removesuffix('.txt') }} {{ 'x'.removesuffix('') }}")
add("methods-unicode-strip", "[{{ '　x '.strip() }}][{{ '​x'.strip() }}]")
add("method-mutating", "{{ msgs.append(1) }}")
add("method-dict-update", "{{ d.update({'a': 1}) }}")
add("list-methods", "{{ [1,2,1].count(1) }} {{ [1,2].index(2) }}")
# Values from the client.
add("client-values", "{{ s }}|{{ n }}|{{ f }}|{{ neg }}|{{ t }}|{{ nothing }}|{{ big }}|{{ fl }}|{{ uni }}")
add("message-loop", "{%- for m in msgs %}<|{{ m.role }}|>{{ m['content'] }}{% if m.reasoning_content is defined %}[{{ m.reasoning_content }}]{% endif %}\n{%- endfor %}")
add("tools-loop", "{%- for t in tools %}{{ t.function.name }}: {{ t.function.parameters.properties|tojson }} {{ t.function.parameters.required|join(',') }}{% endfor %}")
add("nested-dict-iter", "{% for k, v in nested.x|dictsort %}{{ k }}={{ v }}{% endfor %}")
add("is-mapping-content", "{% for m in msgs %}{% if m.content is string %}s{% elif m.content is mapping %}m{% else %}o{% endif %}{% endfor %}")
add("unsupported-include", "{% include 'x' %}", llmp="unsupported")
add("unsupported-call-block", "{% call foo() %}x{% endcall %}", llmp="unsupported")
add("syntax-unclosed", "{% if true %}x")
add("syntax-bad-expr", "{{ 1 + }}")
add("syntax-endfor-alone", "{% endfor %}")
add("break-outside-loop", "{% break %}")
# Edges the independent review checked: rounding, slices with extreme steps,
# searches, strips, reversal, indents and Python's own refusals.
add("review-round", "{{ 2.675|round(2) }}|{{ 0.125|round(2) }}|{{ 3|round }}|{{ 15|round(-1) }}|{{ 25|round(-1) }}|{{ -15|round(-1) }}|{{ 2.5|round }}|{{ 1234.5678|round(-2) }}|{{ 7|round(1, 'ceil') }}|{{ -0.4|round }}")
add("review-sum-strings", "{{ ['a', 'b']|sum(start='') }}")
add("review-undefined-repr", "{{ [nope, 1] }}|{{ nope }}")
add("review-tojson-indent", "{{ {'a': [1, {'b': 2}]}|tojson(indent=70) }}|{{ [1, [2]]|tojson(indent=-1) }}|{{ {'k': [1, 2]}|tojson(separators=(';', '=')) }}")
add("review-indent", "{{ 'a\\nb\\n\\nc'|indent(3) }}|{{ 'a\\nb'|indent('> ', true) }}|{{ 'a\\nb'|indent(-2) }}|{{ 'a\\n\\nb'|indent(2, blank=true) }}")
add("review-slices", "{{ [1, 2, 3, 4, 5][1::9223372036854775807] }}|{{ 'abcdé'[::-2] }}|{{ 'abcdé'[-9223372036854775807:2] }}|{{ [1, 2, 3][::-9223372036854775807] }}|{{ 'héllo'[1:4] }}|{{ [1, 2, 3][5:1:-1] }}|{{ 'héllo'[10:] }}|{{ (1, 2, 3)[::2] }}")
add("review-search", "{{ 'aaa'.rsplit('aa') }}|{{ 'aaa'.split('aa') }}|{{ 'abcabcab'.rfind('cab') }}|{{ 'aaaa'.count('aa') }}|{{ 'abab'.replace('ab', 'x', 1) }}|{{ 'ab' in 'aab' }}|{{ 'aXbXXc'.rsplit('X', 1) }}|{{ 'xabcabc'.find('abcabd') }}|{{ 'ababcabab'.rfind('abab') }}|{{ 'aabaabaaab'.split('aab') }}")
add("review-strip", "[{{ 'xyhixy'.strip('yx') }}][{{ 'héé'.rstrip('é') }}][{{ '  a  '.rstrip() }}][{{ 'ééa'.lstrip('é') }}][{{ ''.strip() }}][{{ '   '.strip() }}][{{ 'a\\u2028'|trim }}]")
add("review-first-last-reverse", "{{ 'héllo'|first }}{{ 'héllö'|last }}|{{ ''|first }}|{{ s|reverse }}|{{ 'é😀a'|reverse }}|{{ [1, 2]|reverse|list }}")
add("review-escape", "{{ '<a & \"b\">'|escape }}|{{ ('<' ~ s ~ '>')|e }}")
add("review-case-tests-any-value", "{{ 5 is lower }} {{ none is upper }} {{ x is lower }} {{ [1] is lower }} {{ ['a'] is lower }} {{ true is upper }} {{ {'a': 1} is lower }} {{ '' is lower }} {{ nothing is lower }}")
add("review-number-underscores", "{{ '1_0'|float }} {{ '_1'|float }} {{ '1_'|float(7) }} {{ '1__0'|float(7) }} {{ 'in_f'|float(7) }} {{ '1_0.5'|float }} {{ '1e1_0'|float }} {{ '1._5'|float(7) }} {{ '1_.5'|int(7) }} {{ '1.5_0'|int(7) }}")
add("review-format-op-mapping", "{{ 'abc' % [] }}|{{ 'abc' % [1] }}|{{ '%s' % [1] }}|{{ '%s' % {'a': 1} }}|{{ 'abc' % {} }}")
add("review-format-op-extra", "{{ 'abc' % 5 }}")
add("review-call-keyword-repeated", "{% set n = namespace(a=1, a=2) %}{{ n.a }}")


def render(template, extra):
    from transformers.utils import chat_template_utils

    class FixedNow(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.datetime(2026, 10, 2, 12, 34, 56)

    chat_template_utils.datetime = FixedNow
    ctx = dict(CTX)
    ctx.update(extra)
    try:
        compiled = chat_template_utils._compile_jinja_template(template)
        return {"text": compiled.render(**ctx)}
    except Exception as e:  # noqa: BLE001 (exceptions are results)
        return {"error": f"{type(e).__name__}: {e}"[:200]}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    import transformers
    out = {"renderer": f"transformers {transformers.__version__} chat-template environment",
           "now": "2026-10-02T12:34:56", "variables": CTX, "snippets": []}
    for s in S:
        entry = {"name": s["name"], "template": s["template"]}
        if s["extra"]:
            entry["variables"] = s["extra"]
        entry.update(render(s["template"], s["extra"]))
        if s["llmp"]:
            entry["llmp"] = s["llmp"]
        out["snippets"].append(entry)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=True)  # ASCII: the variables hold control characters
        f.write("\n")
    print(f"{len(S)} snippets, {sum('error' in e for e in out['snippets'])} errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
