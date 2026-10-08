// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The retained-backing replay (benchmarks/retained_backing/): its trace
// reader and reference, and each candidate design's invariants on small
// synthetic traces, checked after every tick: placements disjoint and
// 4 KiB aligned on held backing, held backing within the budget less any
// outstanding shrink and equal to the fake provider's, the design's
// resident set within the reference's (it evicts the reference's victims
// first), and two replays that agree exactly.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "retained_backing/designs.h"
#include "retained_backing/json.h"
#include "retained_backing/replay.h"
#include "retained_backing/trace.h"

namespace {

using llmp::rb::Call;
using llmp::rb::DesignSpec;
using llmp::rb::Ev;
using llmp::rb::Event;
using llmp::rb::GroupId;
using llmp::rb::HolePolicy;
using llmp::rb::Json;
using llmp::rb::Metrics;
using llmp::rb::Reference;
using llmp::rb::Replay;
using llmp::rb::ReplayOptions;
using llmp::rb::Trace;

constexpr std::uint64_t kMiB = std::uint64_t{1} << 20U;
constexpr std::uint64_t kKiB4 = 4096;

std::uint64_t Count(const Metrics& m, Call call) { return m.calls[static_cast<std::size_t>(call)]; }

// Builds a trace the way swap_trace.py writes one: the reference's evict
// and restore records precede the access or shrink that needs them.
class Builder {
 public:
  explicit Builder(std::uint64_t budget) { trace_.budget = budget; }

  // Groups are added model by model.
  GroupId Add(std::uint32_t model, std::uint64_t stored, bool expert = false) {
    if (trace_.models.size() <= model) {
      trace_.models.push_back(llmp::rb::Model{.name = std::to_string(model),
                                              .first = static_cast<GroupId>(trace_.groups.size()),
                                              .end = static_cast<GroupId>(trace_.groups.size())});
    }
    trace_.groups.push_back(
        llmp::rb::Group{.model = model, .expert = expert, .used = stored, .stored = stored});
    ++trace_.models.back().end;
    return static_cast<GroupId>(trace_.groups.size() - 1);
  }

  std::uint32_t Lease(const std::vector<GroupId>& ids) {
    Push(Event{.ev = Ev::kLease, .lease = next_lease_}, ids);
    return next_lease_++;
  }
  void Release(std::uint32_t lease) { Push(Event{.ev = Ev::kRelease, .lease = lease}, {}); }
  void Use(const std::vector<GroupId>& ids) { Push(Event{.ev = Ev::kUse}, ids); }
  void Shrink(std::uint64_t bytes) { Push(Event{.ev = Ev::kShrink, .bytes = bytes}, {}); }
  void Grow(std::uint64_t bytes) { Push(Event{.ev = Ev::kGrow, .bytes = bytes}, {}); }
  // lease, release
  void Touch(const std::vector<GroupId>& ids) { Release(Lease(ids)); }

  Trace Finish() {
    Reference reference(trace_.groups, trace_.budget);
    Trace out = trace_;
    out.events.clear();
    out.ids.clear();
    for (std::size_t i = 0; i < access_.size(); ++i) {
      const auto& [event, ids] = access_[i];
      auto step = reference.Apply(event, ids);
      EXPECT_TRUE(step.has_value()) << "the reference refuses access " << i;
      if (!step) {
        return out;
      }
      Append(out, Event{.ev = Ev::kEvict}, step->evicted);
      Append(out, Event{.ev = Ev::kRestore}, step->restored);
      Append(out, event, ids, /*always=*/true);
    }
    return out;
  }

  // The index in Finish()'s events of the `n`th access pushed.
  std::size_t IndexOf(std::size_t n) {
    const Trace trace = Finish();
    std::size_t seen = 0;
    for (std::size_t i = 0; i < trace.events.size(); ++i) {
      const Ev ev = trace.events[i].ev;
      if (ev != Ev::kEvict && ev != Ev::kRestore && seen++ == n) {
        return i;
      }
    }
    return SIZE_MAX;
  }
  std::size_t accesses() const { return access_.size(); }

 private:
  void Push(Event event, const std::vector<GroupId>& ids) { access_.emplace_back(event, ids); }

  static void Append(Trace& out, Event event, const std::vector<GroupId>& ids,
                     bool always = false) {
    if (ids.empty() && !always) {
      return;
    }
    event.first = static_cast<std::uint32_t>(out.ids.size());
    event.count = static_cast<std::uint32_t>(ids.size());
    out.ids.insert(out.ids.end(), ids.begin(), ids.end());
    out.events.push_back(event);
  }

  Trace trace_;
  std::vector<std::pair<Event, std::vector<GroupId>>> access_;
  std::uint32_t next_lease_ = 0;
};

// A deterministic generator's stream.
class Lcg {
 public:
  explicit Lcg(std::uint64_t seed) : state_(seed) {}
  std::uint64_t Below(std::uint64_t n) {
    state_ = (state_ * 6364136223846793005ULL) + 1442695040888963407ULL;
    return (state_ >> 33U) % n;
  }

 private:
  std::uint64_t state_;
};

// Three models of dense groups (one spanning several small slabs) and
// uniform experts, in A->B->A episodes with a shrink probe at each B.
Trace RandomTrace(std::uint64_t seed, std::uint64_t budget, std::uint64_t probe) {
  Lcg rng(seed);
  Builder b(budget);
  std::vector<std::vector<GroupId>> dense(3);
  std::vector<std::vector<GroupId>> experts(3);
  for (std::uint32_t m = 0; m < 3; ++m) {
    for (int i = 0; i < 4; ++i) {
      dense[m].push_back(b.Add(m, (1 + rng.Below(2048)) * kKiB4));  // up to 8 MiB
    }
    dense[m].push_back(b.Add(m, (10 * kMiB) + (rng.Below(2560) * kKiB4)));  // 10-20 MiB
    const std::uint64_t expert = (128 + rng.Below(640)) * kKiB4;            // 0.5-3 MiB
    for (int e = 0; e < 24; ++e) {
      experts[m].push_back(b.Add(m, expert, /*expert=*/true));
    }
  }
  auto request = [&](std::uint32_t m) {
    const std::uint32_t lease = b.Lease(dense[m]);
    for (int step = 0; step < 8; ++step) {
      std::vector<GroupId> use;
      for (const GroupId e : experts[m]) {
        if (rng.Below(5) == 0 && use.size() < 6) {
          use.push_back(e);
        }
      }
      if (!use.empty()) {
        b.Use(use);
      }
    }
    b.Release(lease);
  };
  for (int episode = 0; episode < 12; ++episode) {
    const auto a = static_cast<std::uint32_t>(rng.Below(3));
    const auto other = static_cast<std::uint32_t>((a + 1 + rng.Below(2)) % 3);
    request(a);
    b.Shrink(probe);
    request(other);
    b.Grow(probe);
    request(a);
  }
  return b.Finish();
}

std::vector<DesignSpec> TestDesigns() {
  std::vector<DesignSpec> designs{DesignSpec{.name = "d033", .baseline = true}};
  for (const std::uint64_t slab : {8 * kMiB, 32 * kMiB}) {
    for (const HolePolicy policy : {HolePolicy::kContiguousRun, HolePolicy::kSizeClasses,
                                    HolePolicy::kHybrid, HolePolicy::kCompaction}) {
      designs.push_back(DesignSpec{
          .name = std::to_string(slab / kMiB) + "m-" + std::to_string(static_cast<int>(policy)),
          .slab = slab,
          .policy = policy});
    }
  }
  return designs;
}

// ------------------------------------------------------------------ input

TEST(RbJsonTest, ParsesTheTraceSubsetStrictly) {
  const auto value = Json::Parse(R"({"ev":"use","groups":[1,2,3],"x":null,"t":true,"n":-4})");
  ASSERT_TRUE(value.has_value());
  EXPECT_EQ(value->Find("ev")->string(), "use");
  EXPECT_EQ(value->Find("groups")->items().size(), 3U);
  EXPECT_EQ(value->Find("n")->integer(), -4);
  EXPECT_TRUE(value->Find("x")->is_null());
  for (const char* bad :
       {R"({"a":1,"a":2})", R"({"a":1.5})", R"({"a":1e3})", "[1,]", "{} x", R"({"a":01})",
        R"({"a":99999999999999999999})", R"(["\b"])", "[[[[[[[[[[[[[[[[[[1]]]]]]]]]]]]]]]]]"}) {
    EXPECT_FALSE(Json::Parse(bad).has_value()) << bad;
  }
}

TEST(RbTraceTest, ParseChecksRecords) {
  const std::string header =
      R"({"record":"header","format":"llmp-swap-trace","version":1,"role":"primary",)"
      R"("budget_bytes":8388608,"unique_bytes":8192,"models":1,"groups":2,)"
      R"("chunk_bytes":2097152,"file_align":4096})"
      "\n"
      R"({"record":"model","name":"m","profile":"p","layers":null,"groups":[0,2]})"
      "\n"
      R"({"record":"group","id":0,"model":"m","kind":"layer","layer":0,"expert":null,"used":4000,"stored":4096})"
      "\n";
  const std::string group1 =
      R"({"record":"group","id":1,"model":"m","kind":"expert","layer":0,"expert":0,"used":4096,"stored":4096})"
      "\n";
  const auto trace = llmp::rb::ParseTrace(header + group1 +
                                          R"({"ev":"lease","lease":0,"groups":[0,1]})"
                                          "\n"
                                          R"({"ev":"release","lease":0})"
                                          "\n");
  ASSERT_TRUE(trace.has_value()) << trace.error();
  EXPECT_EQ(trace->groups.size(), 2U);
  EXPECT_TRUE(trace->groups[1].expert);
  EXPECT_EQ(trace->events.size(), 2U);
  EXPECT_EQ(trace->Ids(trace->events[0]).size(), 2U);
  // Stored bytes that are not used bytes rounded to 4 KiB, repeated or
  // unknown ids, unknown events and a missing final newline are refused.
  std::string unaligned = group1;
  unaligned.replace(unaligned.find("\"stored\":4096"), 13, "\"stored\":8192");
  EXPECT_FALSE(llmp::rb::ParseTrace(header + unaligned).has_value());
  EXPECT_FALSE(llmp::rb::ParseTrace(header + group1 +
                                    R"({"ev":"use","groups":[1,1]})"
                                    "\n")
                   .has_value());
  EXPECT_FALSE(llmp::rb::ParseTrace(header + group1 +
                                    R"({"ev":"use","groups":[2]})"
                                    "\n")
                   .has_value());
  EXPECT_FALSE(llmp::rb::ParseTrace(header + group1 +
                                    R"({"ev":"page"})"
                                    "\n")
                   .has_value());
  EXPECT_FALSE(llmp::rb::ParseTrace(header + group1 + R"({"ev":"grow","bytes":1})").has_value());
}

TEST(RbTraceTest, LoadRequiresTheRecordedIdentity) {
  const char* scratch = std::getenv("LLMP_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
  ASSERT_NE(scratch, nullptr);
  const std::filesystem::path dir = std::filesystem::path(scratch) / "rb-identity";
  std::filesystem::create_directories(dir);
  std::ofstream(dir / "manifest.json") << R"({"files":{}})" << "\n";
  const auto primary = llmp::rb::LoadTrace(dir.string(), "trace-r5-4.jsonl", "primary");
  ASSERT_FALSE(primary.has_value());
  EXPECT_NE(primary.error().find("not the recorded primary manifest"), std::string::npos);
  EXPECT_FALSE(llmp::rb::LoadTrace(dir.string(), "trace-r5-4.jsonl", "other").has_value());
  std::filesystem::remove_all(dir);
}

// swap_trace.py's reference: LRU over unleased groups, recency by lease,
// use and release, the lower id older within one event.
TEST(RbReferenceTest, EvictsTheLeastRecentUnleasedGroupsFirst) {
  Builder b(8 * kMiB);
  const GroupId a = b.Add(0, 2 * kMiB);
  const GroupId c = b.Add(0, 2 * kMiB);
  const GroupId d = b.Add(0, 2 * kMiB);
  const GroupId e = b.Add(0, 2 * kMiB);
  const GroupId f = b.Add(0, 4 * kMiB);
  b.Use({a, c, d});                         // a is older than c, c than d
  const std::uint32_t held = b.Lease({a});  // a is leased: not a victim
  b.Use({e});
  b.Use({f});  // needs 2 MiB beyond 8: evicts c and d, the oldest unleased
  b.Release(held);
  const Trace trace = b.Finish();
  std::vector<std::vector<GroupId>> evicted;
  for (const Event& event : trace.events) {
    if (event.ev == Ev::kEvict) {
      const auto ids = trace.Ids(event);
      evicted.emplace_back(ids.begin(), ids.end());
    }
  }
  EXPECT_EQ(evicted, (std::vector<std::vector<GroupId>>{{c, d}}));
}

// ------------------------------------------------------------------ designs

class RbDesignTest : public ::testing::TestWithParam<DesignSpec> {};

TEST_P(RbDesignTest, InvariantsHoldAfterEveryTickAndReplaysAgree) {
  constexpr std::uint64_t kBudget = 128 * kMiB;
  for (const std::uint64_t seed : {1U, 2U, 3U}) {
    const Trace trace = RandomTrace(seed, kBudget, 32 * kMiB);
    const Metrics first = Replay(trace, GetParam(), ReplayOptions{.check_every = 1});
    const Metrics second = Replay(trace, GetParam(), ReplayOptions{.check_every = 1});
    EXPECT_EQ(first, second) << "seed " << seed;
    EXPECT_LE(first.held_peak, kBudget);
    if (first.refusals == 0) {
      EXPECT_GT(first.ticks, 0U);
      EXPECT_GE(first.restored_bytes, first.reference_restored_bytes);
      EXPECT_EQ(first.restored_bytes - first.reference_restored_bytes, first.extra_restored_bytes);
    }
    // Size classes and the hybrid give a class its own slabs, which leased
    // dense groups can leave no room for, so they may refuse these traces
    // (so may any design in principle); the others place all of them.
    if (GetParam().baseline || GetParam().policy == HolePolicy::kContiguousRun ||
        GetParam().policy == HolePolicy::kCompaction) {
      EXPECT_EQ(first.refusals, 0U) << "seed " << seed;
    }
    if (!GetParam().baseline) {
      EXPECT_EQ(Count(first, Call::kRegister), Count(first, Call::kCreate));
      EXPECT_EQ(Count(first, Call::kUnregister), Count(first, Call::kRelease));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Designs, RbDesignTest, ::testing::ValuesIn(TestDesigns()),
                         [](const auto& info) {
                           std::string name = info.param.name;
                           std::ranges::replace(name, '-', '_');
                           return name;
                         });

TEST(RbDesignsTest, TheThirteenInTheCriteriaOrder) {
  std::vector<std::string> names;
  for (const DesignSpec& spec : llmp::rb::AllDesigns()) {
    names.push_back(spec.name);
  }
  EXPECT_EQ(names, (std::vector<std::string>{
                       "d033", "slab32m-run", "slab32m-class", "slab32m-hybrid", "slab32m-compact",
                       "slab256m-run", "slab256m-class", "slab256m-hybrid", "slab256m-compact",
                       "slab1g-run", "slab1g-class", "slab1g-hybrid", "slab1g-compact"}));
}

// With groups that fill 2 MiB chunks and slabs exactly, nothing fragments:
// every design evicts the reference's victims and restores what it does.
TEST(RbDesignsTest, WithoutFragmentationEveryDesignRestoresWhatTheReferenceDoes) {
  Lcg rng(7);
  Builder b(64 * kMiB);
  std::vector<GroupId> groups;
  groups.reserve(40);
  for (int i = 0; i < 40; ++i) {
    groups.push_back(b.Add(0, 4 * kMiB));
  }
  for (int i = 0; i < 300; ++i) {
    std::vector<GroupId> use;
    for (int k = 0; k < 3; ++k) {
      const GroupId g = groups[rng.Below(groups.size())];
      if (std::ranges::find(use, g) == use.end()) {
        use.push_back(g);
      }
    }
    std::ranges::sort(use);
    b.Use(use);
  }
  const Trace trace = b.Finish();
  for (const DesignSpec& spec :
       {DesignSpec{.name = "d033", .baseline = true},
        DesignSpec{.name = "run", .slab = 8 * kMiB, .policy = HolePolicy::kContiguousRun},
        DesignSpec{.name = "class", .slab = 8 * kMiB, .policy = HolePolicy::kSizeClasses},
        DesignSpec{.name = "compact", .slab = 32 * kMiB, .policy = HolePolicy::kCompaction}}) {
    const Metrics m = Replay(trace, spec, ReplayOptions{.check_every = 1});
    EXPECT_EQ(m.refusals, 0U) << spec.name;
    EXPECT_EQ(m.content_lost(), 0U) << spec.name;
    EXPECT_EQ(m.restores, m.reference_restores) << spec.name;
    EXPECT_EQ(m.access_extra_evictions, 0U) << spec.name;
    if (spec.baseline) {
      EXPECT_EQ(m.waste_peak, 0U);  // a slab's unfilled tail is waste while the pool fills
    }
  }
}

// D-033: handles freed by an access's evictions go to its restores, and
// what it does not take is released when it ends; no free pool.
TEST(RbDesignsTest, D033HandsHandlesToTheAccessThatEvicted) {
  Builder b(8 * kMiB);
  const GroupId a = b.Add(0, 4 * kMiB);
  const GroupId c = b.Add(0, 4 * kMiB);
  const GroupId d = b.Add(0, 4 * kMiB);
  const GroupId e = b.Add(0, 2 * kMiB);
  b.Touch({a});
  b.Touch({c});
  b.Touch({d});  // evicts a: its two handles become d's
  b.Touch({e});  // evicts c: one handle becomes e's, one is released
  const Metrics m = Replay(b.Finish(), DesignSpec{.name = "d033", .baseline = true},
                           ReplayOptions{.check_every = 1});
  EXPECT_EQ(Count(m, Call::kReserve), 1U);
  EXPECT_EQ(Count(m, Call::kCreate), 4U);
  EXPECT_EQ(Count(m, Call::kMap), 7U);
  EXPECT_EQ(Count(m, Call::kSetAccess), 4U);
  EXPECT_EQ(Count(m, Call::kUnmap), 2U);
  EXPECT_EQ(Count(m, Call::kRelease), 1U);
  EXPECT_EQ(Count(m, Call::kRegister), 0U);
  EXPECT_EQ(m.content_lost(), 0U);
  EXPECT_EQ(m.waste_peak, 0U);
}

// D-033 pays its padding: a group of 2 MiB + 4 KiB holds two handles, so
// where the reference fits three such groups in 8 MiB of stored bytes, the
// baseline evicts beyond it and restores again.
TEST(RbDesignsTest, D033PaddingLosesContentTheReferenceKeeps) {
  Builder b(8 * kMiB);
  const GroupId x = b.Add(0, (2 * kMiB) + kKiB4);
  const GroupId y = b.Add(0, (2 * kMiB) + kKiB4);
  const GroupId z = b.Add(0, (2 * kMiB) + kKiB4);
  b.Touch({x});
  b.Touch({y});
  b.Touch({z});
  b.Touch({x});
  const Metrics m = Replay(b.Finish(), DesignSpec{.name = "d033", .baseline = true},
                           ReplayOptions{.check_every = 1});
  EXPECT_EQ(m.reference_restores, 3U);
  EXPECT_EQ(m.extra_restores, 1U);
  EXPECT_EQ(m.extra_restored_bytes, (2 * kMiB) + kKiB4);
  EXPECT_EQ(m.waste_peak, (2 * (2 * kMiB)) - (2 * kKiB4));
}

// Size classes give each class its own slab, so three leased groups of
// three sizes need three slabs where the budget holds two: a refusal of an
// access the reference admits. The replay stops there.
TEST(RbDesignsTest, SizeClassesCanRefuseWhatTheReferenceAdmits) {
  Builder b(64 * kMiB);
  const GroupId x = b.Add(0, 20 * kMiB);
  const GroupId y = b.Add(0, 18 * kMiB);
  const GroupId z = b.Add(0, 16 * kMiB);
  b.Touch({x, y, z});
  const std::size_t lease = b.IndexOf(0);
  const Trace trace = b.Finish();
  const Metrics classes = Replay(
      trace, DesignSpec{.name = "class", .slab = 32 * kMiB, .policy = HolePolicy::kSizeClasses},
      ReplayOptions{.check_every = 1});
  EXPECT_EQ(classes.refusals, 1U);
  EXPECT_EQ(classes.refused_at, static_cast<std::int64_t>(lease));
  EXPECT_EQ(classes.ticks, 0U);
  const Metrics runs = Replay(
      trace, DesignSpec{.name = "run", .slab = 32 * kMiB, .policy = HolePolicy::kContiguousRun},
      ReplayOptions{.check_every = 1});
  EXPECT_EQ(runs.refusals, 0U);
  const Metrics baseline =
      Replay(trace, DesignSpec{.name = "d033", .baseline = true}, ReplayOptions{.check_every = 1});
  EXPECT_EQ(baseline.refusals, 0U);
}

// A class that needs a slot reclaims a whole slab of another class whose
// groups are all older than its own least recently used group, rather than
// evicting that newer group of its own: extra victims keep the reference's
// recency order across classes.
TEST(RbDesignsTest, SizeClassesReclaimAnOlderSlabOfAnotherClass) {
  Builder b(64 * kMiB);
  const GroupId x1 = b.Add(0, 16 * kMiB);
  const GroupId x2 = b.Add(0, 16 * kMiB);
  std::vector<GroupId> y;
  y.reserve(5);
  for (int i = 0; i < 5; ++i) {
    y.push_back(b.Add(0, 8 * kMiB));
  }
  b.Touch({x1});
  b.Touch({x2});
  for (std::size_t i = 0; i < 4; ++i) {
    b.Touch({y[i]});  // slab 0 holds x1 and x2, slab 1 y[0..3]
  }
  b.Touch({y[4]});  // the reference evicts x1; x2 is older than every y
  b.Touch({y[0]});  // still resident: the design evicted x2, not y[0]
  const Metrics m =
      Replay(b.Finish(),
             DesignSpec{.name = "class", .slab = 32 * kMiB, .policy = HolePolicy::kSizeClasses},
             ReplayOptions{.check_every = 1});
  EXPECT_EQ(m.refusals, 0U);
  EXPECT_EQ(m.access_extra_evicted_bytes, 16 * kMiB);  // x2
  EXPECT_EQ(m.extra_restored_bytes, 0U);
  EXPECT_EQ(Count(m, Call::kCreate), 2U);  // slab 0 passed from x's class to y's
  EXPECT_EQ(Count(m, Call::kRelease), 0U);
}

// In the hybrid, a dense group that needs general space while every slab
// holds a class empties the oldest whole slab, not LRU victims scattered
// over slabs: here one 16 MiB expert where LRU order would take two.
TEST(RbDesignsTest, HybridEmptiesTheOldestSlabForADenseGroup) {
  Builder b(64 * kMiB);
  std::vector<GroupId> e;
  e.reserve(4);
  for (int i = 0; i < 4; ++i) {
    e.push_back(b.Add(0, 8 * kMiB, /*expert=*/true));
  }
  const GroupId f1 = b.Add(0, 16 * kMiB, /*expert=*/true);
  const GroupId f2 = b.Add(0, 16 * kMiB, /*expert=*/true);
  const GroupId dense = b.Add(0, 24 * kMiB);
  for (const GroupId g : {e[0], e[1], f1, e[2], f2, e[3]}) {
    b.Touch({g});  // slab 0: the e experts; slab 1: f1 and f2
  }
  b.Touch({dense});  // the reference evicts e[0], e[1] and f1
  b.Touch({e[2]});   // still resident: the design emptied slab 1 (f2)
  const Metrics m = Replay(
      b.Finish(), DesignSpec{.name = "hybrid", .slab = 32 * kMiB, .policy = HolePolicy::kHybrid},
      ReplayOptions{.check_every = 1});
  EXPECT_EQ(m.refusals, 0U);
  EXPECT_EQ(m.access_extra_evicted_bytes, 16 * kMiB);  // f2
  EXPECT_EQ(m.extra_restored_bytes, 0U);
}

// Compaction moves an unleased dense group to open a hole where
// contiguous-run eviction evicts the coldest run and restores it later.
TEST(RbDesignsTest, CompactionRelocatesWhereRunsEvict) {
  Builder b(64 * kMiB);
  const GroupId a = b.Add(0, 8 * kMiB);
  const GroupId c = b.Add(0, 8 * kMiB);
  const GroupId d = b.Add(0, 8 * kMiB);
  const GroupId big = b.Add(0, 24 * kMiB);
  const GroupId e = b.Add(0, 16 * kMiB);
  const GroupId f = b.Add(0, 16 * kMiB);
  // Slab 0: a c d [big; slab 1: big] e. Then c is touched, so the
  // reference's victims for f are a and d: two 8 MiB holes either side of c.
  b.Touch({a});
  b.Touch({c});
  b.Touch({d});
  b.Touch({big});
  b.Touch({e});
  b.Touch({c});
  b.Touch({f});
  b.Touch({big});
  const Trace trace = b.Finish();
  const Metrics compact = Replay(
      trace, DesignSpec{.name = "compact", .slab = 32 * kMiB, .policy = HolePolicy::kCompaction},
      ReplayOptions{.check_every = 1});
  EXPECT_EQ(compact.relocations, 1U);
  EXPECT_EQ(compact.relocated_bytes, 8 * kMiB);
  EXPECT_EQ(compact.delayed_admissions, 1U);
  EXPECT_EQ(compact.content_lost(), 0U);
  const Metrics runs = Replay(
      trace, DesignSpec{.name = "run", .slab = 32 * kMiB, .policy = HolePolicy::kContiguousRun},
      ReplayOptions{.check_every = 1});
  EXPECT_EQ(runs.relocations, 0U);
  EXPECT_EQ(runs.extra_restored_bytes, 24 * kMiB);  // big, evicted as the coldest run
}

// A shrink probe returns whole slabs: where the reference's victims leave
// each slab partly full, the design evicts more and says so.
TEST(RbDesignsTest, ShrinksReturnWholeSlabs) {
  Builder b(64 * kMiB);
  std::vector<GroupId> groups;
  groups.reserve(8);
  for (int i = 0; i < 8; ++i) {
    groups.push_back(b.Add(0, 8 * kMiB));
  }
  for (const GroupId g : groups) {
    b.Touch({g});
  }
  b.Touch({groups[0], groups[2], groups[4], groups[6]});  // odd ones are now older
  b.Shrink(32 * kMiB);
  b.Grow(32 * kMiB);
  const Metrics m =
      Replay(b.Finish(),
             DesignSpec{.name = "run", .slab = 32 * kMiB, .policy = HolePolicy::kContiguousRun},
             ReplayOptions{.check_every = 1});
  // The reference evicts the four odd groups, half of each slab; one slab
  // must go, so two more groups are evicted beyond the reference.
  EXPECT_EQ(m.shrink_extra_evictions, 2U);
  EXPECT_EQ(m.shrink_extra_evicted_bytes, 16 * kMiB);
  EXPECT_EQ(Count(m, Call::kRelease), 1U);
  EXPECT_EQ(Count(m, Call::kUnregister), 1U);
}

}  // namespace
