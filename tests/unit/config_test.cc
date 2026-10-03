// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The config module: the node document's parsing, merging and schema
// (ParseNodeConfig), and reading it from a main file and its drop-in
// directory (LoadNodeConfig) on scratch trees.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "config/node_config.h"
#include "config/storage_roles.h"
#include "platform/direct_io.h"

namespace {

namespace fs = std::filesystem;
using ::jitllm::config::Diagnostic;
using ::jitllm::config::FormatDiagnostic;
using ::jitllm::config::LoadNodeConfig;
using ::jitllm::config::LoadOptions;
using ::jitllm::config::NodeConfig;
using ::jitllm::config::ParseNodeConfig;
using ::jitllm::config::SourceText;
using ::testing::AllOf;
using ::testing::AnyOf;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::SizeIs;

const fs::path kAnchor = "/var/lib/jitllm/enrollment";

std::vector<std::string> Messages(const std::vector<Diagnostic>& diagnostics) {
  std::vector<std::string> out;
  out.reserve(diagnostics.size());
  for (const Diagnostic& d : diagnostics) {
    out.push_back(FormatDiagnostic(d));
  }
  return out;
}

// The diagnostics of files that must fail, formatted.
std::vector<std::string> Failures(const std::vector<SourceText>& files) {
  auto result = ParseNodeConfig(files, kAnchor);
  EXPECT_FALSE(result.has_value());
  return result ? std::vector<std::string>{} : Messages(result.error());
}

std::vector<std::string> Failures(std::string_view text) {
  return Failures({{"a.toml", std::string(text)}});
}

NodeConfig Parsed(const std::vector<SourceText>& files) {
  auto result = ParseNodeConfig(files, kAnchor);
  EXPECT_TRUE(result.has_value()) << ::testing::PrintToString(result ? std::vector<std::string>{}
                                                                     : Messages(result.error()));
  return result.value_or(NodeConfig{});
}

NodeConfig Parsed(std::string_view text) { return Parsed({{"a.toml", std::string(text)}}); }

constexpr std::string_view kMember = R"(schema_version = 2
cluster_file = "/etc/jitllm/cluster.toml"
node_id = "af564a6b-8b4e-4528-8140-e50b92b40002"

[credentials]
ca_file = "/etc/jitllm/credentials/ca.pem"
certificate_file = "/etc/jitllm/credentials/node.pem"
private_key_file = "/etc/jitllm/credentials/node-key.pem"

[control]
port = 7443
interfaces = "auto"

[limits]
profile = "initial-v2"
)";

TEST(NodeConfigTest, NoFilesAreTheStandaloneDefaults) {
  const NodeConfig config = Parsed(std::vector<SourceText>{});
  EXPECT_FALSE(config.membership.has_value());
  EXPECT_EQ(config.limits_profile, "initial-v2");
  EXPECT_EQ(config.storage.data_dir, "/var/lib/jitllm");
  EXPECT_EQ(config.storage.installed, "/var/lib/jitllm/models");
  EXPECT_EQ(config.storage.spill, "/var/lib/jitllm/spill");
  EXPECT_EQ(config.storage.state, "/var/lib/jitllm/state");
  EXPECT_EQ(config.storage.checkpoints, "/var/lib/jitllm/checkpoints");
  EXPECT_FALSE(config.storage.long_term.has_value());
  EXPECT_FALSE(config.storage.archive.has_value());
  EXPECT_THAT(config.files, IsEmpty());
}

TEST(NodeConfigTest, VersionAloneIsTheDefaults) {
  const NodeConfig config = Parsed("schema_version = 2\n");
  EXPECT_FALSE(config.membership.has_value());
  EXPECT_EQ(config.storage.installed, "/var/lib/jitllm/models");
  EXPECT_THAT(config.files, ElementsAre("a.toml"));
  using Kind = jitllm::config::BindEntry::Kind;
  ASSERT_EQ(config.client.bind.size(), 2U);
  EXPECT_EQ(config.client.bind[0].kind, Kind::kLoopback);
  EXPECT_EQ(config.client.bind[1].kind, Kind::kTailscale);
  EXPECT_EQ(config.client.port, 8114);
  // Every limit permissive by default (D-102): no count, wait or deadline,
  // the byte limits derived from memory, a stall reported only.
  EXPECT_FALSE(config.client.max_connections.has_value());
  EXPECT_FALSE(config.client.max_queued.has_value());
  EXPECT_FALSE(config.client.queue_wait_seconds.has_value());
  EXPECT_EQ(config.client.stall_seconds, 120U);
  EXPECT_EQ(config.client.stall_action, jitllm::config::StallAction::kReport);
  EXPECT_FALSE(config.client.deadline_cap_seconds.has_value());
  EXPECT_EQ(config.client.idle_seconds, 60U);
  EXPECT_EQ(config.client.request_inactivity_seconds, 60U);
  EXPECT_FALSE(config.client.write_inactivity_seconds.has_value());
  EXPECT_FALSE(config.client.hang_seconds.has_value());
  EXPECT_FALSE(config.client.request_memory_bytes.has_value());
  EXPECT_FALSE(config.client.max_body_bytes.has_value());
  EXPECT_FALSE(config.client.stream_buffer_bytes.has_value());
}

// The chat route's listener (D-097 as amended 2026-09-28): symbolic
// entries and any address, one or a list, with the port and the limits.
TEST(NodeConfigTest, TheClientBindsWhereConfigured) {
  using Kind = jitllm::config::BindEntry::Kind;
  const NodeConfig four = Parsed("schema_version = 2\n[client]\nbind = \"127.0.0.2:9000\"\n");
  ASSERT_EQ(four.client.bind.size(), 1U);
  EXPECT_EQ(four.client.bind[0].kind, Kind::kAddress);
  EXPECT_EQ(four.client.bind[0].endpoint.address, "127.0.0.2");
  EXPECT_EQ(four.client.bind[0].endpoint.port, 9000);
  const NodeConfig list = Parsed(
      "schema_version = 2\n[client]\nbind = [\"loopback\", \"[::]\", \"0.0.0.0:80\", "
      "\"[FD7A:115C:A1E0::1]:9\", \"192.168.1.5\"]\nport = 9100\nmax_connections = 1000000\n"
      "max_queued = 1\nstall_seconds = 7200\ndeadline_cap_seconds = 2592000\n"
      "queue_wait_seconds = 1\nstall_action = \"fail\"\nidle_seconds = 5\n"
      "request_inactivity_seconds = 600\nwrite_inactivity_seconds = 30\n"
      "max_body_bytes = 4294967295\nrequest_memory_bytes = 8589934592\n"
      "stream_buffer_bytes = 4096\nhang_seconds = 900\n");
  ASSERT_EQ(list.client.bind.size(), 5U);
  EXPECT_EQ(list.client.bind[0].kind, Kind::kLoopback);
  EXPECT_EQ(list.client.bind[1].endpoint.address, "::");
  EXPECT_TRUE(list.client.bind[1].endpoint.ipv6);
  EXPECT_EQ(list.client.bind[1].endpoint.port, 0);  // [client] port's
  EXPECT_EQ(list.client.bind[2].endpoint.address, "0.0.0.0");
  EXPECT_EQ(list.client.bind[2].endpoint.port, 80);
  EXPECT_EQ(list.client.bind[3].endpoint.address, "fd7a:115c:a1e0::1");
  EXPECT_EQ(list.client.bind[4].endpoint.address, "192.168.1.5");
  EXPECT_EQ(list.client.port, 9100);
  EXPECT_EQ(list.client.max_connections, 1000000U);
  EXPECT_EQ(list.client.max_queued, 1U);
  EXPECT_EQ(list.client.stall_seconds, 7200U);
  EXPECT_EQ(list.client.deadline_cap_seconds, 2592000U);
  EXPECT_EQ(list.client.queue_wait_seconds, 1U);
  EXPECT_EQ(list.client.stall_action, jitllm::config::StallAction::kFail);
  EXPECT_EQ(list.client.idle_seconds, 5U);
  EXPECT_EQ(list.client.request_inactivity_seconds, 600U);
  EXPECT_EQ(list.client.write_inactivity_seconds, 30U);
  EXPECT_EQ(list.client.max_body_bytes, 4294967295U);
  EXPECT_EQ(list.client.request_memory_bytes, 8589934592U);
  EXPECT_EQ(list.client.stream_buffer_bytes, 4096U);
  EXPECT_EQ(list.client.hang_seconds, 900U);
  const NodeConfig six = Parsed("schema_version = 2\nclient.bind = \"[::1]:8114\"\n");
  EXPECT_EQ(six.client.bind[0].endpoint.address, "::1");
  for (const std::string_view bad :
       {"127.0.0.1:0", "127.0.0.1:65536", "127.0.0.1:08114", "localhost:8114", "::1:8114", "::1",
        "127.0.0.1:+80", "[::1]8114", "[::1", "", "Loopback", "tailnet", "1.2.3", "0.0.0.0:"}) {
    EXPECT_THAT(Failures(std::format("schema_version = 2\n[client]\nbind = \"{}\"\n", bad)),
                ElementsAre(HasSubstr("client.bind must be \"loopback\", \"tailscale\" or an")))
        << bad;
  }
  EXPECT_THAT(Failures("schema_version = 2\n[client]\nbind = \"[::ffff:127.0.0.1]:8114\"\n"),
              ElementsAre(HasSubstr("IPv4-mapped")));
  EXPECT_THAT(Failures("schema_version = 2\n[client]\nbind = [\"loopback\", 8114]\n"),
              ElementsAre(HasSubstr("client.bind[1] must be a string, not an integer")));
  EXPECT_THAT(Failures("schema_version = 2\n[client]\nbind = []\n"),
              ElementsAre(HasSubstr("client.bind must hold 1 to 16 entries, not 0")));
  EXPECT_THAT(Failures("schema_version = 2\n[client]\nbind = 8114\n"),
              ElementsAre(HasSubstr("client.bind must be a string or an array of strings")));
  EXPECT_THAT(Failures("schema_version = 2\n[client]\nport = 0\n"),
              ElementsAre(HasSubstr("client.port must be from 1 to 65535, not 0")));
  // Only the types' and arithmetic's ranges remain (D-102).
  EXPECT_THAT(Failures("schema_version = 2\n[client]\nmax_connections = 4294967296\n"),
              ElementsAre(HasSubstr("client.max_connections must be from 1 to 4294967295")));
  EXPECT_THAT(Failures("schema_version = 2\n[client]\nmax_queued = 0\n"),
              ElementsAre(HasSubstr("client.max_queued must be from 1 to 4294967295")));
  EXPECT_THAT(Failures("schema_version = 2\n[client]\nmax_queued = \"4\"\n"),
              ElementsAre(HasSubstr("client.max_queued must be an integer")));
  EXPECT_TRUE(Parsed("schema_version = 2\n[client]\nstall_seconds = 1\n").client.stall_seconds ==
              1U);
  EXPECT_THAT(Failures("schema_version = 2\n[client]\nstall_seconds = 0\n"),
              ElementsAre(HasSubstr("client.stall_seconds must be from 1 to 2592000 seconds")));
  EXPECT_THAT(Failures("schema_version = 2\n[client]\nstall_seconds = 2592001\n"),
              ElementsAre(HasSubstr("client.stall_seconds must be from 1 to 2592000 seconds")));
  EXPECT_THAT(Failures("schema_version = 2\n[client]\ndeadline_cap_seconds = 0\n"),
              ElementsAre(HasSubstr("client.deadline_cap_seconds must be from 1 to 2592000")));
  EXPECT_THAT(Failures("schema_version = 2\n[client]\nstall_seconds = 120.0\n"),
              ElementsAre(HasSubstr("client.stall_seconds must be an integer, not a float")));
  EXPECT_THAT(Failures("schema_version = 2\n[client]\nstall_action = \"restart\"\n"),
              ElementsAre(HasSubstr(R"(client.stall_action must be "report" or "fail")")));
  EXPECT_THAT(Failures("schema_version = 2\n[client]\nmax_body_bytes = 4294967296\n"),
              ElementsAre(HasSubstr("client.max_body_bytes must be from 1024 to 4294967295")));
  EXPECT_THAT(Failures("schema_version = 2\n[client]\nstream_buffer_bytes = 4095\n"),
              ElementsAre(HasSubstr("client.stream_buffer_bytes must be from 4096")));
  EXPECT_THAT(Failures("schema_version = 2\n[client]\nrequest_memory_bytes = 1048576\n"),
              ElementsAre(HasSubstr("client.request_memory_bytes must be from 16777216")));
  EXPECT_THAT(Failures("schema_version = 2\n[client]\nmax_body_bytes = 33554432\n"
                       "request_memory_bytes = 16777216\n"),
              ElementsAre(HasSubstr("client.max_body_bytes (33554432) must be at most "
                                    "client.request_memory_bytes (16777216)")));
  EXPECT_THAT(Failures("schema_version = 2\n[client]\nhang_seconds = 59\n"),
              ElementsAre(HasSubstr("client.hang_seconds must be from 60 to 2592000 seconds")));
  // The removed keys are unknown now.
  EXPECT_THAT(Failures("schema_version = 2\n[client]\nbody_budget_bytes = 1048576\n"),
              ElementsAre(HasSubstr("unknown key client.body_budget_bytes")));
  EXPECT_THAT(Failures("schema_version = 2\n[client]\nhosts = []\n"),
              ElementsAre(HasSubstr("unknown key client.hosts")));
}

// Conversation state under memory pressure (D-055 as amended 2026-10-02):
// its retention and the spill budget, with their defaults and ranges.
TEST(NodeConfigTest, MemoryKeepsConversationsAsConfigured) {
  const NodeConfig defaults = Parsed("schema_version = 2\n");
  EXPECT_EQ(defaults.memory.retention_hours, 24U);
  EXPECT_EQ(defaults.memory.spill_budget_gib, 128U);
  const NodeConfig set =
      Parsed("schema_version = 2\n[memory]\nretention_hours = 1\nspill_budget_gib = 0\n");
  EXPECT_EQ(set.memory.retention_hours, 1U);
  EXPECT_EQ(set.memory.spill_budget_gib, 0U);
  const NodeConfig most = Parsed(
      "schema_version = 2\nmemory.retention_hours = 8760\nmemory.spill_budget_gib = "
      "1048576\n");
  EXPECT_EQ(most.memory.retention_hours, 8760U);
  EXPECT_EQ(most.memory.spill_budget_gib, 1048576U);
  EXPECT_THAT(Failures("schema_version = 2\n[memory]\nretention_hours = 0\n"),
              ElementsAre(HasSubstr("memory.retention_hours must be from 1 to 8760, not 0")));
  EXPECT_THAT(Failures("schema_version = 2\n[memory]\nretention_hours = 8761\n"),
              ElementsAre(HasSubstr("memory.retention_hours must be from 1 to 8760")));
  EXPECT_THAT(Failures("schema_version = 2\n[memory]\nspill_budget_gib = -1\n"),
              ElementsAre(HasSubstr("memory.spill_budget_gib must be from 0 to 1048576")));
  EXPECT_THAT(Failures("schema_version = 2\n[memory]\nspill_budget_gib = \"64\"\n"),
              ElementsAre(HasSubstr("memory.spill_budget_gib must be an integer")));
  EXPECT_THAT(Failures("schema_version = 2\n[memory]\nplans = 3\n"),
              ElementsAre(HasSubstr("unknown key memory.plans")));
}

TEST(NodeConfigTest, ReadsTheMemberExample) {
  const NodeConfig config = Parsed(kMember);
  ASSERT_TRUE(config.membership.has_value());
  const jitllm::config::Membership member =
      config.membership.value_or(jitllm::config::Membership{});
  EXPECT_EQ(member.cluster_file, "/etc/jitllm/cluster.toml");
  EXPECT_EQ(member.node_id, "af564a6b-8b4e-4528-8140-e50b92b40002");
  EXPECT_EQ(member.credentials.private_key_file, "/etc/jitllm/credentials/node-key.pem");
  EXPECT_EQ(member.control.port, 7443);
  EXPECT_THAT(member.control.interfaces, IsEmpty());
}

TEST(NodeConfigTest, ReadsSelectorsAndPeerScopes) {
  const NodeConfig config = Parsed(std::string(kMember).replace(
      std::string(kMember).find("interfaces = \"auto\""), 19,
      "interfaces = [\"ifname:enp1s0f0np0\", \"port:0c42a1b2/p0\"]\n"
      "[control.peer_scopes]\n"
      "af564a6b-8b4e-4528-8140-e50b92b40003 = \"ifname:enp1s0f1np1\""));
  ASSERT_TRUE(config.membership.has_value());
  const jitllm::config::Membership member =
      config.membership.value_or(jitllm::config::Membership{});
  EXPECT_THAT(member.control.interfaces, ElementsAre("ifname:enp1s0f0np0", "port:0c42a1b2/p0"));
  EXPECT_EQ(member.control.peer_scopes.at("af564a6b-8b4e-4528-8140-e50b92b40003"),
            "ifname:enp1s0f1np1");
}

TEST(NodeConfigTest, ResolvesStorageRoles) {
  const NodeConfig config = Parsed(R"(schema_version = 2
[storage]
data_dir = "/srv/jitllm"
installed = "/nvme/models"
long_term = "/mnt/nas/jitllm"
checkpoints = "/mnt/nas/jitllm/checkpoints"
)");
  EXPECT_EQ(config.storage.installed, "/nvme/models");
  EXPECT_EQ(config.storage.spill, "/srv/jitllm/spill");
  EXPECT_EQ(config.storage.checkpoints, "/mnt/nas/jitllm/checkpoints");
  EXPECT_EQ(config.storage.archive, fs::path("/mnt/nas/jitllm/archive"));
}

TEST(NodeConfigTest, SyntaxErrorsNameTheFileAndPosition) {
  EXPECT_THAT(Failures("schema_version = 2\nstorage = [\n"), ElementsAre(HasSubstr("a.toml:2:")));
  EXPECT_THAT(Failures("schema_version = 2\na = 1\na = 2\n"), ElementsAre(HasSubstr("a.toml:3:")));
  EXPECT_THAT(Failures("schema_version = 2\nk = \"\xff\"\n"), ElementsAre(HasSubstr("utf-8")));
}

// What the toml++ pin was chosen for, and cluster-design.md's integer rules.
TEST(NodeConfigTest, TheParserIsStrict) {
  const std::string base = "schema_version = 2\n";
  for (const std::string& bad : {
           base + "[control]\nport = 07443\n",                 // leading zero
           base + "[control]\nport = 99999999999999999999\n",  // beyond 64 bits
           base + "[storage]\n[storage]\n",                    // a table defined twice
           base + "[storage]\nspill = \"a\x01b\"\n",           // a raw control character
           base + "a = " + std::string(5000, '[') + std::string(5000, ']') + "\n",  // nesting
           base + "a" +
               [] {
                 std::string keys;
                 for (int i = 0; i < 2000; ++i) {
                   keys += ".a";  // dotted keys past toml++'s depth limit
                 }
                 return keys;
               }() +
               " = 1\n",
           base + "[[[a]]]\n",
       }) {
    EXPECT_FALSE(ParseNodeConfig(std::vector<SourceText>{{"a.toml", bad}}, kAnchor).has_value())
        << bad.substr(0, 60);
  }
}

// A deep table the schema does not know is one problem, not one per level,
// and the report is bounded.
TEST(NodeConfigTest, UnknownDepthCostsOneProblem) {
  std::string deep = "schema_version = 2\n";
  for (int line = 0; line < 200; ++line) {
    deep += std::format("[b{}", line);
    for (int level = 0; level < 1000; ++level) {
      deep += ".a";
    }
    deep += "]\n";
  }
  const std::vector<std::string> failures = Failures(deep);
  ASSERT_THAT(failures, SizeIs(jitllm::config::kDiagnosticLimit + 1));
  EXPECT_THAT(failures.front(), HasSubstr("a.toml:2:1: unknown table b0"));
  EXPECT_EQ(failures.back(), "and 100 more problems");
}

TEST(NodeConfigTest, InlineTablesAreCheckedLikeHeaders) {
  const NodeConfig config =
      Parsed("schema_version = 2\nstorage = { spill = \"s\", data_dir = \"/d\" }\n");
  EXPECT_EQ(config.storage.spill, "/d/s");
  EXPECT_THAT(Failures("schema_version = 2\nstorage = { spil = \"s\" }\n"),
              ElementsAre(HasSubstr("unknown key storage.spil")));
  EXPECT_THAT(Failures("schema_version = 2\ncontrol = {}\n"),
              Contains(HasSubstr("node_id is required")));
}

TEST(NodeConfigTest, AnEmptyMemberTableMakesAMember) {
  EXPECT_THAT(Failures("schema_version = 2\n[control]\n"),
              Contains(HasSubstr("cluster_file is required")));
  EXPECT_THAT(Failures("schema_version = 2\n[credentials]\n"),
              Contains(HasSubstr("control.port is required")));
}

TEST(NodeConfigTest, DiagnosticsCannotForgeLines) {
  std::string text(kMember);
  text.replace(text.find(R"("auto")"), 6, R"("x\njitllm-runtime: ready\u202e")");
  const std::vector<std::string> failures = Failures(text);
  ASSERT_THAT(failures, SizeIs(1));
  EXPECT_THAT(failures[0], HasSubstr("x\\x0ajitllm-runtime: ready\\u202e"));
  EXPECT_THAT(
      Failures("schema_version = 2\n[storage]\nspill = \"a\\u202eb\"\n"),
      ElementsAre(HasSubstr("must not contain control or invisible formatting characters")));
}

TEST(NodeConfigTest, EveryFileNeedsTheSupportedVersion) {
  EXPECT_THAT(Failures("[storage]\n"), Contains(HasSubstr("schema_version is missing")));
  EXPECT_THAT(Failures("schema_version = 1\n"),
              ElementsAre(HasSubstr("schema_version 1 is not supported")));
  EXPECT_THAT(Failures("schema_version = \"2\"\n"), ElementsAre(HasSubstr("must be an integer")));
  EXPECT_THAT(
      Failures({{"a.toml", "schema_version = 2\n"}, {"b.toml", "[storage]\nspill = \"s\"\n"}}),
      ElementsAre("b.toml: schema_version is missing; every file of the configuration starts with "
                  "schema_version = 2"));
}

TEST(NodeConfigTest, ReportsEveryProblemNotJustTheFirst) {
  const std::vector<std::string> failures = Failures({
      {"a.toml", "schema_version = 2\nnode_idd = 1\n[storage]\nspill = 5\n"},
      {"b.toml", "schema_version = 2\n[limits]\nprofile = \"big\"\n"},
      {"c.toml", "schema_version = 2\n[[broken\n"},
  });
  EXPECT_THAT(failures, SizeIs(4));
  EXPECT_THAT(failures, Contains(HasSubstr("a.toml:2:12: unknown key node_idd")));
  EXPECT_THAT(
      failures,
      Contains(HasSubstr("a.toml:4:9: storage.spill must be a string path, not an integer")));
  EXPECT_THAT(failures, Contains(HasSubstr("b.toml:3:11: limits.profile must be \"initial-v2\"")));
  EXPECT_THAT(failures, Contains(HasSubstr("c.toml:2:")));
}

TEST(NodeConfigTest, UnknownKeysAndTablesAreFatal) {
  EXPECT_THAT(Failures("schema_version = 2\n[storage]\nspil = \"x\"\n"),
              ElementsAre(HasSubstr("unknown key storage.spil")));
  // [client] has bind, port and the limits (D-097, D-102); the front door's
  // other keys arrive in M5.
  EXPECT_THAT(Failures("schema_version = 2\n[client]\ncredential = \"x\"\n"),
              ElementsAre(HasSubstr("unknown key client.credential")));
  EXPECT_THAT(Failures("schema_version = 2\nstorage = \"x\"\n"),
              ElementsAre(HasSubstr("storage must be a table")));
  EXPECT_THAT(Failures("schema_version = 2\n[storage.spill]\n"),
              ElementsAre(HasSubstr("storage.spill must be a path, not a table")));
  EXPECT_THAT(Failures("schema_version = 2\n\"a.b\" = 1\n"),
              ElementsAre(HasSubstr("unknown key \"a.b\"")));
}

TEST(NodeConfigTest, EachKeyHasOneOwningFile) {
  EXPECT_THAT(Failures({{"a.toml", "schema_version = 2\n[storage]\nspill = \"s\"\n"},
                        {"b.toml", "schema_version = 2\nstorage.spill = \"t\"\n"}}),
              ElementsAre(HasSubstr("b.toml:2:17: storage.spill is also set in a.toml")));
  // An inline table is one value, owned whole.
  EXPECT_THAT(
      Failures({{"a.toml", "schema_version = 2\nstorage = { spill = \"s\" }\n"},
                {"b.toml", "schema_version = 2\n[storage]\ninstalled = \"m\"\n"}}),
      AllOf(Contains(HasSubstr("storage.installed is also set in a.toml (as storage)")),
            Contains(HasSubstr("b.toml:2:1: storage is a table here, but a.toml sets storage"))));
  // Tables merge across files.
  const NodeConfig config =
      Parsed({{"a.toml", "schema_version = 2\n[storage]\nspill = \"s\"\n"},
              {"b.toml", "schema_version = 2\n[storage]\ninstalled = \"m\"\n"}});
  EXPECT_EQ(config.storage.spill, "/var/lib/jitllm/s");
  EXPECT_EQ(config.storage.installed, "/var/lib/jitllm/m");
}

TEST(NodeConfigTest, MembersNeedEveryMemberKey) {
  const std::vector<std::string> failures =
      Failures("schema_version = 2\n[control]\nport = 7443\n");
  EXPECT_THAT(failures, SizeIs(5));
  EXPECT_THAT(failures, Contains(HasSubstr("cluster_file is required")));
  EXPECT_THAT(failures, Contains(HasSubstr("credentials.private_key_file is required")));
}

TEST(NodeConfigTest, ChecksMemberValues) {
  const auto with = [](std::string_view from, std::string_view to) {
    std::string text(kMember);
    text.replace(text.find(from), from.size(), to);
    return Failures(text);
  };
  EXPECT_THAT(with("af564a6b-8b4e-4528-8140-e50b92b40002", "AF564A6B-8B4E-4528-8140-E50B92B40002"),
              ElementsAre(HasSubstr("node_id must be a canonical lowercase UUID")));
  EXPECT_THAT(with("port = 7443", "port = 0"),
              ElementsAre(HasSubstr("control.port must be from 1 to 65535, not 0")));
  EXPECT_THAT(with("port = 7443", "port = 65536"), ElementsAre(HasSubstr("not 65536")));
  EXPECT_THAT(with("port = 7443", "port = 7443.0"),
              ElementsAre(HasSubstr("must be an integer, not a float")));
  EXPECT_THAT(with("\"auto\"", "\"all\""), ElementsAre(HasSubstr("must be \"auto\" or an array")));
  EXPECT_THAT(with("\"auto\"", "[]"), ElementsAre(HasSubstr("must not be an empty array")));
  EXPECT_THAT(with("\"auto\"", "[\"eth0\"]"), ElementsAre(HasSubstr("must be \"ifname:<name>\"")));
  EXPECT_THAT(with("\"auto\"", "[\"ifname:a/b\"]"), ElementsAre(HasSubstr("invalid interface")));
  EXPECT_THAT(with("\"auto\"", "[\"ifname:0123456789abcdef\"]"),
              ElementsAre(HasSubstr("invalid interface")));
  EXPECT_THAT(with("\"auto\"", "[\"port:XY/p0\"]"), ElementsAre(HasSubstr("invalid port")));
  EXPECT_THAT(with("\"auto\"", "[\"ifname:a\", \"ifname:a\"]"),
              ElementsAre(HasSubstr("lists \"ifname:a\" twice")));
  EXPECT_THAT(with("\"/etc/jitllm/cluster.toml\"", "\"cluster.toml\""),
              ElementsAre(HasSubstr("cluster_file must be an absolute path")));
  EXPECT_THAT(with("interfaces = \"auto\"", "peer_scopes = { nope = \"ifname:a\" }"),
              ElementsAre(HasSubstr("the key must be a member's canonical lowercase UUID")));
}

TEST(NodeConfigTest, PathsAreInNormalForm) {
  for (const std::string_view bad :
       {R"("")", R"("a//b")", R"("a/./b")", R"("../b")", R"("a/")", R"("a\u0001b")"}) {
    EXPECT_THAT(
        Failures(std::string("schema_version = 2\n[storage]\nspill = ") + std::string(bad) + "\n"),
        ElementsAre(HasSubstr("storage.spill")))
        << bad;
  }
  EXPECT_THAT(Failures("schema_version = 2\n[storage]\ndata_dir = \"var/lib\"\n"),
              ElementsAre(HasSubstr("storage.data_dir must be an absolute path")));
}

TEST(NodeConfigTest, RolesNeverOverlap) {
  EXPECT_THAT(Failures("schema_version = 2\n[storage]\nspill = \"models\"\n"),
              ElementsAre(HasSubstr("storage.installed and storage.spill are the same directory")));
  EXPECT_THAT(Failures("schema_version = 2\n[storage]\nspill = \"models/spill\"\n"),
              ElementsAre(HasSubstr(
                  "storage.spill (/var/lib/jitllm/models/spill) lies inside storage.installed")));
  EXPECT_THAT(Failures("schema_version = 2\n[storage]\ninstalled = \"/var/lib\"\n"),
              Contains(HasSubstr("lies inside storage.installed")));
  // long_term never touches a runtime role, but may hold the job-only ones.
  EXPECT_THAT(Failures("schema_version = 2\n[storage]\nlong_term = \"/var/lib/jitllm/state/lt\"\n"),
              Contains(HasSubstr(
                  "storage.long_term (/var/lib/jitllm/state/lt) must not equal, contain or lie "
                  "inside storage.state")));
  EXPECT_THAT(Failures("schema_version = 2\n[storage]\nlong_term = \"/var/lib\"\n"),
              Contains(HasSubstr("must not equal, contain or lie inside storage.installed")));
  EXPECT_THAT(Failures("schema_version = 2\n[storage]\narchive = \"a\"\n"),
              ElementsAre(HasSubstr("storage.archive requires storage.long_term")));
  EXPECT_THAT(
      Failures("schema_version = 2\n[storage]\nlong_term = \"/nas\"\narchive = \"/elsewhere\"\n"),
      ElementsAre(HasSubstr("must lie inside storage.long_term")));
  EXPECT_THAT(Failures("schema_version = 2\n[storage]\nlong_term = \"/nas\"\narchive = \"/nas\"\n"),
              ElementsAre(HasSubstr("must lie inside storage.long_term")));
  EXPECT_THAT(
      Failures("schema_version = 2\n[storage]\nlong_term = \"/nas\"\ncheckpoints = \"/nas\"\n"),
      ElementsAre(HasSubstr("storage.archive (/nas/archive) lies inside storage.checkpoints")));
}

TEST(NodeConfigTest, RolesStayClearOfTheAnchor) {
  EXPECT_THAT(Failures("schema_version = 2\n[storage]\nstate = \"enrollment\"\n"),
              ElementsAre(HasSubstr(
                  "storage.state (/var/lib/jitllm/enrollment) must not equal, contain or lie "
                  "inside the enrollment anchor")));
  EXPECT_THAT(Failures("schema_version = 2\n[storage]\nstate = \"/var/lib/jitllm\"\n"),
              Contains(HasSubstr("storage.state (/var/lib/jitllm) must not equal, contain")));
  EXPECT_THAT(Failures("schema_version = 2\n[storage]\nstate = \"enrollment/x\"\n"),
              ElementsAre(HasSubstr("the enrollment anchor")));
}

TEST(NodeConfigTest, MemberFilesStayOutOfTheLongTermStore) {
  std::string text(kMember);
  text += "[storage]\nlong_term = \"/etc/jitllm/credentials\"\n";
  EXPECT_THAT(Failures(text),
              AllOf(SizeIs(3), Contains(HasSubstr("credentials.ca_file must not lie inside "
                                                  "storage.long_term"))));
}

TEST(NodeConfigTest, TheDocumentHasASizeLimit) {
  std::string big = "schema_version = 2\n# ";
  big.append(std::size_t{1} << 20, 'x');
  EXPECT_THAT(Failures(big), ElementsAre(HasSubstr("more than the 1048576-byte limit")));
}

// D-096's models: a table each, sorted by name, with their defaults.
constexpr std::string_view kDsv4 =
    "8a355bfb27c90e1150fbd7fa62ea6e63f6bf34fcca33934e52d22773f1508234";
constexpr std::string_view kDspark =
    "dd2d3f9c66f070fb231d27d5a11f38ff22c78dc8f089cecedbb67721e9b4bec5";
constexpr std::string_view kImage =
    "eca21baad38229e471a44cb2479d392ffcf745fb812e8a41668f336139fa1acd";

TEST(NodeConfigTest, ReadsModels) {
  const NodeConfig config = Parsed(std::format(R"(schema_version = 2
[models."qwen3.8"]
artifact = "{0}"
speculation = false
context = 4096
tokenizer = "/opt/qwen/tokenizer.json"
chat_template = "/opt/qwen/chat_template.jinja"

[models.deepseek]
artifact = "{1}"
drafter = "{2}"

[models]
image = {{ composition = "{3}" }}
)",
                                               std::string(64, 'a'), kDsv4, kDspark, kImage));
  ASSERT_THAT(config.models, SizeIs(3));
  const auto& deepseek = config.models[0];
  EXPECT_EQ(deepseek.name, "deepseek");
  EXPECT_EQ(deepseek.artifact, std::string(kDsv4));
  EXPECT_EQ(deepseek.drafter, std::string(kDspark));
  // Settings not set are not overrides: registration resolves them (D-103).
  EXPECT_TRUE(deepseek.overrides.empty());
  EXPECT_FALSE(deepseek.Bool("speculation").has_value());
  EXPECT_FALSE(deepseek.Integer("context").has_value());
  EXPECT_FALSE(deepseek.composition.has_value());
  const auto& image = config.models[1];
  EXPECT_EQ(image.name, "image");
  EXPECT_EQ(image.composition, std::string(kImage));
  EXPECT_FALSE(image.artifact.has_value());
  const auto& qwen = config.models[2];
  EXPECT_EQ(qwen.name, "qwen3.8");
  EXPECT_EQ(qwen.Bool("speculation"), false);
  EXPECT_EQ(qwen.Integer("context"), 4096);
  EXPECT_THAT(qwen.overrides, SizeIs(2));
  EXPECT_EQ(qwen.tokenizer, fs::path("/opt/qwen/tokenizer.json"));
  EXPECT_EQ(qwen.chat_template, fs::path("/opt/qwen/chat_template.jinja"));
}

TEST(NodeConfigTest, ChecksModels) {
  const std::string id(64, 'b');
  const auto failures = Failures(std::format(
      R"(schema_version = 2
[models.Upper]
artifact = "{0}"
[models.none]
context = 4096
[models.both]
artifact = "{1}"
composition = "{2}"
[models.pipeline]
composition = "{3}"
drafter = "{4}"
[models.short]
artifact = "abc"
[models.small]
artifact = "{5}"
context = 16
[models.self]
artifact = "{6}"
drafter = "{6}"
[models.again]
artifact = "{1}"
[models.odd]
artifact = "{7}"
tokenizer = "relative/tokenizer.json"
colour = "blue"
[models.deep.er]
artifact = "{0}"
)",
      id, std::string(64, 'c'), std::string(64, 'd'), std::string(64, 'e'), std::string(64, 'f'),
      std::string(64, '1'), std::string(64, '2'), std::string(64, '3')));
  EXPECT_THAT(failures, Contains(HasSubstr("models.Upper: a model's name is 1-64 characters")));
  EXPECT_THAT(failures, Contains(HasSubstr("models.none: a model names exactly one of")));
  EXPECT_THAT(failures, Contains(HasSubstr("models.both: a model names exactly one of")));
  EXPECT_THAT(failures, Contains(HasSubstr("models.pipeline.drafter is a model artifact's key, "
                                           "not a composition's")));
  EXPECT_THAT(failures, Contains(HasSubstr("models.short.artifact must be an ID string")));
  EXPECT_THAT(failures, Contains(HasSubstr("models.small.context must be from 512 to 4294967295")));
  EXPECT_THAT(failures, Contains(HasSubstr("models.self: an artifact cannot be its own drafter")));
  EXPECT_THAT(failures, Contains(HasSubstr("an installed artifact serves one model")));
  EXPECT_THAT(failures, Contains(HasSubstr("models.odd.tokenizer must be an absolute path")));
  EXPECT_THAT(failures, Contains(HasSubstr("unknown key models.odd.colour")));
  EXPECT_THAT(failures, Contains(HasSubstr("unknown table models.deep.er")));
}

// A speculative model's wave form: auto by default, or forced for exactness
// controls; only an artifact's key.
TEST(NodeConfigTest, ReadsAndChecksWaveForm) {
  const NodeConfig config = Parsed(std::format(R"(schema_version = 2
[models.chosen]
artifact = "{0}"
[models.auto]
artifact = "{1}"
wave_form = "auto"
[models.drafted]
artifact = "{2}"
wave_form = "speculative"
[models.plain]
artifact = "{3}"
wave_form = "plain"
)",
                                               std::string(64, 'a'), std::string(64, 'b'),
                                               std::string(64, 'c'), std::string(64, 'd')));
  ASSERT_THAT(config.models, SizeIs(4));
  EXPECT_EQ(config.models[0].name, "auto");
  EXPECT_EQ(config.models[0].Text("wave_form"), "auto");
  EXPECT_EQ(config.models[1].name, "chosen");
  EXPECT_FALSE(config.models[1].Text("wave_form").has_value());
  EXPECT_EQ(config.models[2].name, "drafted");
  EXPECT_EQ(config.models[2].Text("wave_form"), "speculative");
  EXPECT_EQ(config.models[3].name, "plain");
  EXPECT_EQ(config.models[3].Text("wave_form"), "plain");

  const auto failures = Failures(std::format(
      R"(schema_version = 2
[models.word]
artifact = "{0}"
wave_form = "fast"
[models.flag]
artifact = "{1}"
wave_form = true
[models.pipeline]
composition = "{2}"
wave_form = "plain"
)",
      std::string(64, 'a'), std::string(64, 'b'), std::string(64, 'c')));
  EXPECT_THAT(
      failures,
      Contains(HasSubstr(R"(models.word.wave_form must be "auto", "speculative" or "plain")")));
  EXPECT_THAT(
      failures,
      Contains(HasSubstr(R"(models.flag.wave_form must be "auto", "speculative" or "plain")")));
  EXPECT_THAT(failures, Contains(HasSubstr("models.pipeline.wave_form is a model artifact's key, "
                                           "not a composition's")));
}

// No generic context ceiling (D-102): the configuration takes any 32-bit
// count from the least; registration checks the checkpoint's own
// (model_limits.h, runtime_test).
TEST(NodeConfigTest, ContextHasNoGenericCeiling) {
  for (const auto context : {512U, 262144U, 262145U, 1048576U, 1048577U, 4294967295U}) {
    const NodeConfig config =
        Parsed(std::format("schema_version = 2\n[models.m]\nartifact = \"{}\"\ncontext = {}\n",
                           std::string(64, 'a'), context));
    ASSERT_THAT(config.models, SizeIs(1));
    EXPECT_EQ(config.models.front().Integer("context"), context);
  }
  EXPECT_THAT(
      Failures(
          std::format("schema_version = 2\n[models.m]\nartifact = \"{}\"\ncontext = 4294967296\n",
                      std::string(64, 'a'))),
      ElementsAre(HasSubstr("context must be from 512 to 4294967295 tokens, not 4294967296")));
}

// The model library may exceed memory (D-102): no count of models.
TEST(NodeConfigTest, NamesAnyNumberOfModels) {
  std::string text = "schema_version = 2\n";
  for (int i = 0; i < 40; ++i) {
    text += std::format("[models.m{}]\nartifact = \"{:064x}\"\n", i, i + 1);
  }
  EXPECT_THAT(Parsed(text).models, SizeIs(40));
}

// A model's context from its minimum, its prefill chunk (configured, or
// absent for the runtime's default) and its throughput floors
// (docs/runtime-serving.md).
TEST(NodeConfigTest, ReadsContextAndPrefillChunk) {
  const NodeConfig config = Parsed(std::format(R"(schema_version = 2
[models.a]
artifact = "{0}"
context = 512
prefill_chunk = 4096
prefill_floor_tok_s = 250
decode_floor_tok_s = 1
[models.b]
artifact = "{1}"
)",
                                               std::string(64, 'a'), std::string(64, 'b')));
  ASSERT_THAT(config.models, SizeIs(2));
  EXPECT_EQ(config.models[0].Integer("context"), jitllm::config::kMinContext);
  EXPECT_EQ(config.models[0].Integer("prefill_chunk"), 4096);
  EXPECT_FALSE(config.models[1].Integer("prefill_chunk").has_value());
  EXPECT_EQ(config.models[0].Integer("prefill_floor_tok_s"), 250);
  EXPECT_EQ(config.models[0].Integer("decode_floor_tok_s"), 1);
  EXPECT_FALSE(config.models[1].Integer("prefill_floor_tok_s").has_value());
  EXPECT_FALSE(config.models[1].Integer("decode_floor_tok_s").has_value());

  const auto failures = Failures(std::format(
      R"(schema_version = 2
[models.below]
artifact = "{0}"
context = 511
[models.none]
artifact = "{1}"
prefill_chunk = 0
[models.over]
artifact = "{2}"
prefill_chunk = 262145
[models.text]
artifact = "{3}"
prefill_chunk = "4096"
[models.pipeline]
composition = "{4}"
prefill_chunk = 512
[models.slow]
artifact = "{5}"
prefill_floor_tok_s = 0
decode_floor_tok_s = 100001
[models.fast]
artifact = "{6}"
prefill_floor_tok_s = 1000001
decode_floor_tok_s = 2.5
[models.picture]
composition = "{7}"
decode_floor_tok_s = 5
)",
      std::string(64, 'a'), std::string(64, 'b'), std::string(64, 'c'), std::string(64, 'd'),
      std::string(64, 'e'), std::string(64, 'f'), std::string(64, '0'), std::string(64, '1')));
  EXPECT_THAT(failures, Contains(HasSubstr("models.below.context must be from 512 to 4294967295")));
  EXPECT_THAT(failures, Contains(HasSubstr("models.none.prefill_chunk must be from 1 to 262144")));
  EXPECT_THAT(failures, Contains(HasSubstr("models.over.prefill_chunk must be from 1 to 262144")));
  EXPECT_THAT(failures, Contains(HasSubstr("models.text.prefill_chunk must be an integer")));
  EXPECT_THAT(failures, Contains(HasSubstr("models.pipeline.prefill_chunk is a model artifact's "
                                           "key, not a composition's")));
  EXPECT_THAT(failures, Contains(HasSubstr("models.slow.prefill_floor_tok_s must be from 1 to "
                                           "1000000 tokens a second, not 0")));
  EXPECT_THAT(failures, Contains(HasSubstr("models.slow.decode_floor_tok_s must be from 1 to "
                                           "100000 tokens a second")));
  EXPECT_THAT(failures, Contains(HasSubstr("models.fast.prefill_floor_tok_s must be from 1 to")));
  EXPECT_THAT(failures, Contains(HasSubstr("models.fast.decode_floor_tok_s must be an integer")));
  EXPECT_THAT(failures, Contains(HasSubstr("models.picture.decode_floor_tok_s is a model "
                                           "artifact's key")));
}

// A model's request slots (docs/runtime-serving.md#request-slots): absent,
// the model's own default; configured, 1 to 16, and only on an artifact.
TEST(NodeConfigTest, ReadsMaxSlots) {
  const NodeConfig config =
      Parsed(std::format(R"(schema_version = 2
[models.a]
artifact = "{0}"
max_slots = 1
[models.b]
artifact = "{1}"
max_slots = 16
[models.c]
artifact = "{2}"
)",
                         std::string(64, 'a'), std::string(64, 'b'), std::string(64, 'c')));
  ASSERT_THAT(config.models, SizeIs(3));
  EXPECT_EQ(config.models[0].Integer("max_slots"), 1);
  EXPECT_EQ(config.models[1].Integer("max_slots"), jitllm::config::kMaxModelSlots);
  EXPECT_FALSE(config.models[2].Integer("max_slots").has_value());

  const auto failures = Failures(std::format(
      R"(schema_version = 2
[models.none]
artifact = "{0}"
max_slots = 0
[models.over]
artifact = "{1}"
max_slots = 17
[models.text]
artifact = "{2}"
max_slots = "4"
[models.pipeline]
composition = "{3}"
max_slots = 2
)",
      std::string(64, 'a'), std::string(64, 'b'), std::string(64, 'c'), std::string(64, 'd')));
  EXPECT_THAT(
      failures,
      Contains(HasSubstr("models.none.max_slots must be from 1 to 16 request slots, not 0")));
  EXPECT_THAT(failures, Contains(HasSubstr("models.over.max_slots must be from 1 to 16")));
  EXPECT_THAT(failures, Contains(HasSubstr("models.text.max_slots must be an integer")));
  EXPECT_THAT(failures, Contains(HasSubstr("models.pipeline.max_slots is a model artifact's key")));
}

// D-103's table-driven schema: a valid value of every key (its least, or
// its first choice) is kept for a model of its kind and refused on the
// other kind; past its range each is refused, naming the key.
std::string ValidValue(const jitllm::config::ModelKeySpec& spec) {
  using jitllm::config::ModelKeyType;
  switch (spec.type) {
    case ModelKeyType::kId:
      return std::format("\"{}\"", std::string(64, '9'));
    case ModelKeyType::kPath:
      return "\"/opt/model/file\"";
    case ModelKeyType::kBool:
      return "false";
    case ModelKeyType::kInteger:
      return std::format("{}", spec.least);
    case ModelKeyType::kReal:
      return std::format("{:.6f}", spec.real_most);
    case ModelKeyType::kChoice:
      return std::format("\"{}\"", spec.choices.back());
    case ModelKeyType::kReals:
      return std::format("[{:.6f}, {:.6f}]", spec.real_most, spec.real_most);
    case ModelKeyType::kText:
      return "\"<x>\"";
  }
  return "0";
}

TEST(NodeConfigTest, EveryModelKeyIsTableDriven) {
  using jitllm::config::kArtifactModels;
  using jitllm::config::kCompositionModels;
  using jitllm::config::ModelKeys;
  using jitllm::config::ModelKeyType;
  for (const auto& spec : ModelKeys()) {
    if (spec.key == "artifact" || spec.key == "composition") {
      continue;
    }
    SCOPED_TRACE(spec.key);
    EXPECT_EQ(jitllm::config::FindModelKey(spec.key), &spec);
    EXPECT_FALSE(spec.summary.empty());
    const std::string value = ValidValue(spec);
    for (const bool artifact : {true, false}) {
      const std::string text =
          std::format("schema_version = 2\n[models.m]\n{} = \"{}\"\n{} = {}\n",
                      artifact ? "artifact" : "composition", std::string(64, 'a'), spec.key, value);
      const std::vector<SourceText> files = {{.name = "a.toml", .text = text}};
      const auto parsed = ParseNodeConfig(files, kAnchor);
      const bool takes = (spec.kinds & (artifact ? kArtifactModels : kCompositionModels)) != 0;
      ASSERT_EQ(parsed.has_value(), takes) << text;
      if (!parsed) {
        EXPECT_THAT(Messages(parsed.error()),
                    Contains(HasSubstr(std::format("models.m.{} is a", spec.key))));
        continue;
      }
      ASSERT_THAT(parsed->models, SizeIs(1));
      const auto& m = parsed->models[0];
      // An identity key fills its field; a setting is an override.
      EXPECT_EQ(m.overrides.contains(spec.key), spec.setting);
      if (spec.type == ModelKeyType::kInteger) {
        EXPECT_EQ(m.Integer(spec.key), spec.least);
      } else if (spec.type == ModelKeyType::kReal) {
        EXPECT_EQ(m.Real(spec.key), spec.real_most);
      } else if (spec.type == ModelKeyType::kReals) {
        EXPECT_EQ(m.Reals(spec.key), (std::vector<double>{spec.real_most, spec.real_most}));
      } else if (spec.type == ModelKeyType::kBool) {
        EXPECT_EQ(m.Bool(spec.key), false);
      }
    }
    // Past its range, or of another type.
    std::vector<std::string> bad;
    switch (spec.type) {
      case ModelKeyType::kInteger:
        bad = {std::format("{}", spec.most + 1), "\"1\"", "1.5"};
        if (spec.least > 0) {
          bad.push_back(std::format("{}", spec.least - 1));
        }
        break;
      case ModelKeyType::kReal:
        bad = {std::format("{:.6f}", spec.real_most + 1), "\"1\"", "nan"};
        break;
      case ModelKeyType::kReals: {
        std::string many = "[1";  // one more than it holds
        for (std::size_t i = 0; i < spec.most_items; ++i) {
          many += ", 1";
        }
        bad = {"[]", std::format("[{:.6f}]", spec.real_most + 1), "1.0", "[\"1\"]", many + "]"};
        break;
      }
      case ModelKeyType::kChoice:
        bad = {"\"neither\"", "true"};
        break;
      case ModelKeyType::kBool:
        bad = {"1", "\"true\""};
        break;
      case ModelKeyType::kText:
        bad = {std::format("\"{}\"", std::string(static_cast<std::size_t>(spec.most) + 1, 'x')),
               R"("a\u0007b")", "1"};
        break;
      case ModelKeyType::kId:
        bad = {"\"abc\"", "1"};
        break;
      case ModelKeyType::kPath:
        bad = {"\"relative/path\"", "1"};
        break;
    }
    const std::string kind = (spec.kinds & kArtifactModels) != 0 ? "artifact" : "composition";
    for (const std::string& v : bad) {
      const auto failures =
          Failures(std::format("schema_version = 2\n[models.m]\n{} = \"{}\"\n{} = {}\n", kind,
                               std::string(64, 'a'), spec.key, v));
      EXPECT_THAT(failures, Contains(HasSubstr(std::format("models.m.{}", spec.key)))) << v;
    }
  }
}

TEST(NodeConfigTest, ChecksModelKeyRules) {
  const auto failures = Failures(std::format(R"(schema_version = 2
[models.a]
artifact = "{0}"
wave_read_align = 3000
[models.b]
artifact = "{1}"
wave_costs = [1.5, 0, 2.9]
top_p = 0
[models.c]
composition = "{2}"
image_size = 1000
[models.d]
artifact = "{3}"
colour = "blue"
)",
                                             std::string(64, 'a'), std::string(64, 'b'),
                                             std::string(64, 'c'), std::string(64, 'd')));
  EXPECT_THAT(failures, Contains(HasSubstr("models.a.wave_read_align must be a power of two")));
  EXPECT_THAT(failures, Contains(HasSubstr("models.b.top_p must be greater than 0 and at most 1")));
  EXPECT_THAT(failures, Contains(HasSubstr("models.c.image_size must be a multiple of 32")));
  EXPECT_THAT(failures, Contains(HasSubstr("unknown key models.d.colour")));
  EXPECT_THAT(failures, Not(Contains(HasSubstr("models.b.wave_costs"))));
  // Applicability to an architecture is registration's: the schema takes
  // a DeepSeek setting on any artifact.
  const NodeConfig config =
      Parsed(std::format("schema_version = 2\n[models.q]\nartifact = \"{}\"\nwave_costs = [2, "
                         "2.5]\ntemperature = 0.6\nreasoning_start = \"\"\n",
                         std::string(64, 'a')));
  ASSERT_THAT(config.models, SizeIs(1));
  EXPECT_EQ(config.models[0].Reals("wave_costs"), (std::vector<double>{2.0, 2.5}));
  EXPECT_EQ(config.models[0].Real("temperature"), 0.6);
  EXPECT_EQ(config.models[0].Text("reasoning_start"), "");
  EXPECT_FALSE(config.models[0].Integer("temperature").has_value());  // of its key's type only
}

TEST(NodeConfigTest, ModelsAreOwnedOnce) {
  // Any number of models (NamesAnyNumberOfModels); a model's table in two
  // files merges; one key set twice does not.
  const std::string a = std::format("schema_version = 2\n[models.x]\nartifact = \"{}\"\n", kDsv4);
  const NodeConfig merged = Parsed(
      {{"a.toml", a},
       {"b.toml", std::format("schema_version = 2\n[models.x]\ndrafter = \"{}\"\n", kDspark)}});
  ASSERT_THAT(merged.models, SizeIs(1));
  EXPECT_EQ(merged.models[0].drafter, std::string(kDspark));
  EXPECT_THAT(Failures({{"a.toml", a}, {"b.toml", a}}),
              Contains(HasSubstr("each key belongs to exactly one file")));
}

// Where scratch trees go: JITLLM_TEST_SCRATCH, in the build tree, so that
// no other user shares their parent directories as they may in /tmp.
fs::path Scratch() {
  const char* scratch = std::getenv("JITLLM_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
  return scratch != nullptr ? fs::path(scratch) : fs::path(::testing::TempDir());
}

class Tree {
 public:
  Tree() {
    std::string pattern = (Scratch() / "config-XXXXXX").string();
    std::error_code error;
    fs::create_directories(Scratch(), error);
    if (::mkdtemp(pattern.data()) != nullptr) {
      path_ = pattern;
    } else {
      ADD_FAILURE() << "cannot create a directory from " << pattern;
    }
  }
  Tree(const Tree&) = delete;
  Tree& operator=(const Tree&) = delete;
  Tree(Tree&&) = delete;
  Tree& operator=(Tree&&) = delete;
  ~Tree() {
    std::error_code error;
    fs::remove_all(path_, error);
  }

  fs::path operator/(std::string_view name) const { return path_ / name; }

  void Write(std::string_view name, std::string_view text, mode_t mode = 0644) const {
    const fs::path file = path_ / name;
    std::error_code error;
    fs::create_directories(file.parent_path(), error);
    std::ofstream(file, std::ios::binary) << text;
    ASSERT_EQ(::chmod(file.c_str(), mode), 0);
  }

  LoadOptions Options(std::string_view main = "jitllm.toml") const {
    return {.main_file = path_ / main,
            .main_file_optional = false,
            .anchor = kAnchor,
            .trusted_uid = ::geteuid()};
  }

 private:
  fs::path path_;
};

std::vector<std::string> LoadFailures(const LoadOptions& options) {
  auto result = LoadNodeConfig(options);
  EXPECT_FALSE(result.has_value());
  return result ? std::vector<std::string>{} : Messages(result.error());
}

TEST(LoadNodeConfigTest, ReadsTheMainFileThenFragmentsInOrder) {
  const Tree tree;
  tree.Write("jitllm.toml", "schema_version = 2\n[storage]\ndata_dir = \"/srv/j\"\n");
  tree.Write("jitllm.d/20-spill.toml", "schema_version = 2\n[storage]\nspill = \"s\"\n");
  tree.Write("jitllm.d/10-enroll.toml", "schema_version = 2\n[limits]\nprofile = \"initial-v2\"\n");
  tree.Write("jitllm.d/.hidden.toml", "not toml at all [");
  tree.Write("jitllm.d/30.toml.dpkg-old", "not toml at all [");
  tree.Write("jitllm.d/README", "not toml at all [");
  auto result = LoadNodeConfig(tree.Options());
  ASSERT_TRUE(result.has_value()) << ::testing::PrintToString(Messages(result.error()));
  EXPECT_THAT(result->files, ElementsAre(tree / "jitllm.toml", tree / "jitllm.d/10-enroll.toml",
                                         tree / "jitllm.d/20-spill.toml"));
  EXPECT_EQ(result->storage.spill, "/srv/j/s");
}

TEST(LoadNodeConfigTest, FragmentsAloneFormTheDocument) {
  const Tree tree;
  tree.Write("jitllm.d/a.toml", "schema_version = 2\n[storage]\nspill = \"s\"\n");
  LoadOptions options = tree.Options();
  EXPECT_THAT(LoadFailures(options), ElementsAre(HasSubstr("jitllm.toml: does not exist")));
  options.main_file_optional = true;
  auto result = LoadNodeConfig(options);
  ASSERT_TRUE(result.has_value());
  EXPECT_THAT(result->files, ElementsAre(tree / "jitllm.d/a.toml"));
}

TEST(LoadNodeConfigTest, NothingAtTheDefaultIsTheDefaults) {
  const Tree tree;
  LoadOptions options = tree.Options();
  options.main_file_optional = true;
  auto result = LoadNodeConfig(options);
  ASSERT_TRUE(result.has_value());
  EXPECT_THAT(result->files, IsEmpty());
  EXPECT_EQ(result->storage.installed, "/var/lib/jitllm/models");
}

TEST(LoadNodeConfigTest, TheMainFileEndsInToml) {
  const Tree tree;
  EXPECT_THAT(LoadFailures(tree.Options("jitllm.conf")),
              ElementsAre(HasSubstr("must end in .toml")));
}

TEST(LoadNodeConfigTest, RefusesLinksAndOtherFiles) {
  const Tree tree;
  tree.Write("real.toml", "schema_version = 2\n");
  fs::create_symlink(tree / "real.toml", tree / "jitllm.toml");
  EXPECT_THAT(LoadFailures(tree.Options()), ElementsAre(HasSubstr("is a symbolic link")));
  fs::remove(tree / "jitllm.toml");
  fs::create_directory(tree / "jitllm.toml");
  EXPECT_THAT(LoadFailures(tree.Options()), ElementsAre(HasSubstr("is not a regular file")));
  fs::remove(tree / "jitllm.toml");
  tree.Write("jitllm.toml", "schema_version = 2\n");
  fs::create_directories(tree / "jitllm.d");
  fs::create_symlink(tree / "real.toml", tree / "jitllm.d/a.toml");
  EXPECT_THAT(LoadFailures(tree.Options()),
              ElementsAre(HasSubstr("jitllm.d/a.toml: is a symbolic link")));
  fs::remove(tree / "jitllm.d/a.toml");
  fs::create_directory(tree / "jitllm.d/b.toml");
  EXPECT_THAT(LoadFailures(tree.Options()),
              ElementsAre(HasSubstr("jitllm.d/b.toml: is not a regular file")));
}

TEST(LoadNodeConfigTest, RefusesWhatOthersCouldChange) {
  const Tree tree;
  // Group write through a group others may share counts as theirs; the
  // private-group case depends on this host's accounts, so only other
  // write is tested here.
  tree.Write("jitllm.toml", "schema_version = 2\n", 0646);
  EXPECT_THAT(LoadFailures(tree.Options()),
              ElementsAre(HasSubstr("can be written by users other than root and its owner")));
  ASSERT_EQ(::chmod((tree / "jitllm.toml").c_str(), 0644), 0);
  tree.Write("jitllm.d/a.toml", "schema_version = 2\n", 0646);
  EXPECT_THAT(LoadFailures(tree.Options()),
              ElementsAre(HasSubstr("jitllm.d/a.toml: can be written")));
  ASSERT_EQ(::chmod((tree / "jitllm.d/a.toml").c_str(), 0644), 0);
  ASSERT_EQ(::chmod((tree / "jitllm.d").c_str(), 01777), 0);
  EXPECT_THAT(LoadFailures(tree.Options()), ElementsAre(HasSubstr("can add files to")));
  ASSERT_EQ(::chmod((tree / "jitllm.d").c_str(), 0755), 0);
  ASSERT_EQ(::chmod((tree / "").c_str(), 0757), 0);
  EXPECT_THAT(LoadFailures(tree.Options()),
              Contains(HasSubstr("can be changed by users other than")));
  ASSERT_EQ(::chmod((tree / "").c_str(), 0700), 0);
  // Owned by another user: trust only root.
  LoadOptions options = tree.Options();
  if (::geteuid() != 0) {
    options.trusted_uid = 0;
    // (Under qemu-user the walk stops at "/", the sysroot this user owns.)
    EXPECT_THAT(LoadFailures(options),
                Contains(AnyOf(HasSubstr("not root"), HasSubstr("is not owned by root"))));
  }
}

TEST(LoadNodeConfigTest, TheNumberOfFragmentsIsBounded) {
  const Tree tree;
  tree.Write("jitllm.toml", "schema_version = 2\n");
  for (std::size_t i = 0; i <= jitllm::config::kDropInLimit; ++i) {
    tree.Write(std::format("jitllm.d/{:04}.toml", i), "schema_version = 2\n");
  }
  EXPECT_THAT(LoadFailures(tree.Options()),
              ElementsAre(HasSubstr("holds 257 fragments; the configuration "
                                    "takes at most 256")));
}

TEST(LoadNodeConfigTest, RefusesHardLinkedFiles) {
  const Tree tree;
  tree.Write("jitllm.toml", "schema_version = 2\n");
  fs::create_hard_link(tree / "jitllm.toml", tree / "other");
  EXPECT_THAT(LoadFailures(tree.Options()), ElementsAre(HasSubstr("has 2 hard links")));
}

// Group write through the owner's private group is the owner's own, but not
// when an ACL grants it to someone else.
TEST(LoadNodeConfigTest, RefusesAnAclGrantingWrite) {
  const Tree tree;
  tree.Write("jitllm.toml", "schema_version = 2\n");
  const std::string command =
      std::format("setfacl -m u:nobody:rw '{}' 2>/dev/null", (tree / "jitllm.toml").string());
  // NOLINTNEXTLINE(concurrency-mt-unsafe,cert-env33-c,bugprone-command-processor): a fixed command
  if (std::system(command.c_str()) != 0) {
    GTEST_SKIP() << "setfacl is not available here";
  }
  EXPECT_THAT(LoadFailures(tree.Options()),
              ElementsAre(HasSubstr("can be written by users other than root")));
}

TEST(LoadNodeConfigTest, FailuresInOneFileDoNotHideTheOthers) {
  const Tree tree;
  tree.Write("jitllm.toml", "schema_version = 2\nbad = 1\n");
  tree.Write("jitllm.d/a.toml", "schema_version = 2\n", 0666);
  tree.Write("jitllm.d/b.toml", "schema_version = 3\n");
  // Files that cannot be trusted are not parsed at all.
  EXPECT_THAT(LoadFailures(tree.Options()),
              ElementsAre(HasSubstr("jitllm.d/a.toml: can be written")));
  ASSERT_EQ(::chmod((tree / "jitllm.d/a.toml").c_str(), 0644), 0);
  EXPECT_THAT(LoadFailures(tree.Options()),
              ElementsAre(HasSubstr("jitllm.toml:2:7: unknown key bad"),
                          HasSubstr("b.toml:1:18: schema_version 3")));
}

// The runtime's roles on a scratch tree; the direct-I/O probe needs the
// build tree on a filesystem the roles accept.
class Roles : public ::testing::Test {
 protected:
  void SetUp() override {
    auto filesystem = jitllm::platform::DescribeFilesystem(tree_ / "");
    ASSERT_TRUE(filesystem.has_value());
    if (!filesystem->accepted) {
      GTEST_SKIP() << "the build tree is on " << filesystem->type
                   << ", which the storage roles refuse";
    }
  }

  jitllm::config::Storage Storage() const {
    return {.data_dir = tree_ / "data",
            .installed = tree_ / "data/models",
            .spill = tree_ / "data/spill",
            .state = tree_ / "data/state",
            .checkpoints = tree_ / "data/checkpoints",
            .long_term = std::nullopt,
            .archive = std::nullopt};
  }

  std::vector<std::string> Problems(const jitllm::config::Storage& storage) const {
    auto result = jitllm::config::PrepareRuntimeRoles(storage, ::geteuid(), tree_ / "enrollment");
    EXPECT_FALSE(result.has_value());
    return result ? std::vector<std::string>{} : result.error();
  }

  static mode_t Mode(const fs::path& path) {
    struct stat status{};
    EXPECT_EQ(::lstat(path.c_str(), &status), 0) << path;
    return status.st_mode & 07777U;
  }

  Tree tree_;
};

TEST_F(Roles, CreatesMissingRolesWithTheirModes) {
  auto roles = jitllm::config::PrepareRuntimeRoles(Storage(), ::geteuid(), tree_ / "enrollment");
  ASSERT_TRUE(roles.has_value()) << ::testing::PrintToString(roles.error());
  EXPECT_EQ(roles->installed, tree_ / "data/models");
  EXPECT_EQ(Mode(tree_ / "data"), 0755U);
  EXPECT_EQ(Mode(tree_ / "data/models"), 0755U);
  EXPECT_EQ(Mode(tree_ / "data/spill"), 0700U);
  EXPECT_EQ(Mode(tree_ / "data/state"), 0700U);
  EXPECT_TRUE(fs::is_regular_file(tree_ / "data/spill/.jitllm-spill"));
  // The runtime never creates the job-only roles.
  EXPECT_FALSE(fs::exists(tree_ / "data/checkpoints"));
  // A second start finds them as it left them.
  roles = jitllm::config::PrepareRuntimeRoles(Storage(), ::geteuid(), tree_ / "enrollment");
  EXPECT_TRUE(roles.has_value()) << ::testing::PrintToString(roles.error());
}

TEST_F(Roles, RefusesASpillDirectoryItDidNotMake) {
  fs::create_directories(tree_ / "data/spill");
  ASSERT_EQ(::chmod((tree_ / "data/spill").c_str(), 0700), 0);
  tree_.Write("data/spill/photos.jpg", "precious");
  EXPECT_THAT(Problems(Storage()),
              ElementsAre(HasSubstr("is not empty and has no .jitllm-spill marker")));
  EXPECT_TRUE(fs::exists(tree_ / "data/spill/photos.jpg"));
  ASSERT_EQ(::chmod((tree_ / "data/spill").c_str(), 0750), 0);
  EXPECT_THAT(Problems(Storage()),
              ElementsAre(HasSubstr("storage.spill (" + (tree_ / "data/spill").string() +
                                    ") has mode 0750, not 0700")));
}

TEST_F(Roles, RefusesLinksAndWritableRoles) {
  fs::create_directories(tree_ / "elsewhere");
  fs::create_directories(tree_ / "data");
  fs::create_directory_symlink(tree_ / "elsewhere", tree_ / "data/state");
  EXPECT_THAT(Problems(Storage()),
              ElementsAre(HasSubstr("storage.state (" + (tree_ / "data/state").string() +
                                    "): is a symbolic link")));
  fs::remove(tree_ / "data/state");
  fs::create_directories(tree_ / "data/models");
  ASSERT_EQ(::chmod((tree_ / "data/models").c_str(), 0757), 0);
  EXPECT_THAT(Problems(Storage()),
              ElementsAre(HasSubstr("storage.installed (" + (tree_ / "data/models").string() +
                                    ") can be written by users other than root")));
}

TEST_F(Roles, NoRoleResolvesOntoTheAnchor) {
  fs::create_directories(tree_ / "real");
  fs::create_directory_symlink(tree_ / "real", tree_ / "dd");
  jitllm::config::Storage storage = Storage();
  storage.state = tree_ / "dd/enrollment";
  auto result =
      jitllm::config::PrepareRuntimeRoles(storage, ::geteuid(), tree_ / "real/enrollment");
  ASSERT_FALSE(result.has_value());
  EXPECT_THAT(result.error(), ElementsAre(HasSubstr("lies inside the enrollment anchor")));
  EXPECT_FALSE(fs::exists(tree_ / "real/enrollment"));
  EXPECT_FALSE(fs::exists(tree_ / "data"));  // nothing is created before every check passes
}

TEST_F(Roles, RefusesAnAnchorOthersCouldReplace) {
  fs::create_directories(tree_ / "open");
  ASSERT_EQ(::chmod((tree_ / "open").c_str(), 0777), 0);
  auto result =
      jitllm::config::PrepareRuntimeRoles(Storage(), ::geteuid(), tree_ / "open/x/enrollment");
  ASSERT_FALSE(result.has_value());
  EXPECT_THAT(result.error(), ElementsAre(HasSubstr("the enrollment anchor")));
  EXPECT_FALSE(fs::exists(tree_ / "data"));
}

TEST_F(Roles, RefusesADefaultAcl) {
  fs::create_directories(tree_ / "data/models");
  const std::string command =
      std::format("setfacl -d -m u:nobody:rwx '{}' 2>/dev/null", (tree_ / "data/models").string());
  // NOLINTNEXTLINE(concurrency-mt-unsafe,cert-env33-c,bugprone-command-processor): a fixed command
  if (std::system(command.c_str()) != 0) {
    GTEST_SKIP() << "setfacl is not available here";
  }
  EXPECT_THAT(Problems(Storage()), ElementsAre(HasSubstr("has a default ACL")));
}

TEST_F(Roles, RefusesAMarkerItDidNotWrite) {
  fs::create_directories(tree_ / "data/spill");
  ASSERT_EQ(::chmod((tree_ / "data/spill").c_str(), 0700), 0);
  tree_.Write("data/spill/.jitllm-spill", "someone else's\n", 0600);
  EXPECT_THAT(Problems(Storage()), ElementsAre(HasSubstr("is not the marker this runtime writes")));
}

TEST_F(Roles, LinksCannotMakeTwoRolesOne) {
  fs::create_directories(tree_ / "real");
  fs::create_directory_symlink(tree_ / "real", tree_ / "alias");
  jitllm::config::Storage storage = Storage();
  storage.installed = tree_ / "real/x";
  storage.spill = tree_ / "alias/x";
  EXPECT_THAT(Problems(storage),
              ElementsAre(HasSubstr("storage.installed and storage.spill are the same directory")));
  EXPECT_FALSE(fs::exists(tree_ / "real/x"));  // refused before anything is created
  storage = Storage();
  storage.spill = tree_ / "alias/s";
  storage.checkpoints = tree_ / "real/s/checkpoints";
  EXPECT_THAT(Problems(storage), ElementsAre(HasSubstr("storage.checkpoints (" +
                                                       (tree_ / "real/s/checkpoints").string() +
                                                       ") overlaps storage.spill")));
}

}  // namespace
