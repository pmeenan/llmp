<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Initial cluster: detected links, configured authority

D-038 with D-039 defines the M6a cluster design, extending [D-037's ownership
contract](architecture.md#conductor-ownership-and-admission). This is an
experimental design for implementation, not a working configuration loader,
discovery service or cluster runtime. Whole-model placement precedes general
sharding (M8); M4's sharded full swaps run earlier on a minimal configured
two-node conductor, before this design is built ([plan](plan.md)).

## Setup experience

Start setup on the node that should provide the client endpoint. It proposes
that node as conductor, inventories its interfaces, and finds candidate peer
addresses on the selected interconnects. Already trusted administrative SSH
access can collect the other nodes' inventories without hand-entering every
interface. With jitLLM setup running on peers, a bounded link-local discovery
window provides candidates too. Setup groups multiple addresses by verified
node identity and proposes a node/link diagram. The user enrolls the desired
nodes and accepts the generated cluster configuration once. Subsequent starts
re-detect paths for those members; they do not require repeating enrollment.
Manual endpoint/interface overrides and a single-node setup remain supported.
Single-node mode opens no remote control/discovery listener; local dispatch
still follows the same admission contract. For an enrolled cluster, background
path refresh uses authenticated inventory updates, configured DNS/seeds and
current neighbor hints. If every endpoint becomes unreachable and no hint
resolves, rerun the bounded setup discovery window to refresh addresses for
existing identities; no automatic membership changes follow. Setup also runs
the bounded dedicated-QSFP subnet scan below, enabled by default in schema v2;
normal serving never launches a sweep.

Detecting a connected QSFP port is automatic. Discovering an address is not
proof of its owner, membership, conductor authority, or permission to receive
prompts. No IP-subnet convention, MAC suffix, hostname ordering or count of
interfaces chooses those things. A fresh setup proposes the initiating node
as conductor rather than electing whichever peer responds first. If setup
finds multiple peers, it proposes the matching topology class below and shows
the candidates. Contradictory or incomplete evidence remains explicitly partial.

The owner requested interface-based initial-layout detection on 2026-09-22.
This brings **bootstrap discovery and ongoing path detection for enrolled
members** into M6a. Automatic membership changes, conductor election and
automatic replicas remain deferred; D-023's original blanket discovery
deferral is narrowed by D-038. D-039 adds the owner-requested single, pair,
triangle and switched-N layout classifier and dedicated-QSFP subnet scanning.

## Detecting the network

The runtime's native network inventory uses Linux interface/address/route
notifications and sysfs/provider metadata. The setup tool may use equivalent
OS tools. The core consumes node, adapter, physical-port and path identities;
NVIDIA names do not enter scheduling policy. A versioned Spark hardware
profile interprets validated physical metadata; unknown hardware gets generic
interface discovery and explicit overrides, not guessed Spark wiring.

1. Enumerate interfaces, addresses (including IPv6 scope), carrier, MTU,
   negotiated speed, driver, PCI identity and RDMA-to-netdev mappings.
   Exclude loopback, container/virtual interfaces and down links from automatic
   interconnect selection. Exclusion is a selection rule, not permission to
   change those devices. Inventory down physical ports for the setup diagram.
2. On the validated Spark profile, group paths by **local adapter identity
   and physical port**. Read `phys_switch_id` plus `phys_port_name`, corroborate
   with devlink physical port and PCI device identity. The two PCI domains
   feeding one port are separate host paths sharing that port's line rate.
   Interface names, `dev_port` alone and link speed alone are insufficient.
   Do not sum their advertised speeds or treat them as independent cable
   failure domains. If grouping metadata is absent/inconsistent, report
   unknown shared capacity and require explicit grouping before multi-path
   bandwidth claims; ordinary authenticated single-path TCP can still work.
3. Candidate addresses come from explicit seeds, existing routes/neighbors,
   an explicitly supplied NVIDIA Sync network plan, or a setup-only mDNS
   service `_jitllm._tcp.local.` on selected interfaces. The plan and neighbor
   cache are hints, including when installed by an administrator. Setup scans
   directly attached IPv4 subnets of detected QSFP ports under the dedicated
   cluster-network assumption, with the scope and bounds below. No discovery-
   triggered remote command execution, Internet discovery or modification of
   addresses/routes/MTU/bonds/firewall. No dependency on reading privileged
   Netplan files; permission denial leaves discovery usable from live data.
4. For mDNS, advertise only an opaque candidate ID, protocol version and
   control port; no model list, prompts or state identifiers. Browse/respond
   only during an explicitly initiated setup window on selected interfaces,
   with TTL/hop-limit and interface scoping checked. Discovery is unauthenticated
   and never enrolls a peer. Missing multicast or disabled IPv6 falls back to
   neighbor hints, trusted SSH inventory or explicit addresses.
5. After enrollment, establish mutually authenticated control connections and
   compare the peer's own inventory. Group addresses by its enrolled node ID,
   not by MAC similarity. Validate candidate routes and source addresses in
   both directions, with interface scope for link-local IPv6. Prefer a live
   authenticated high-speed interconnect path; retain an explicitly permitted
   management-network fallback. Bind a selected source/interface and check the
   actual socket route; an address must not silently traverse a default gateway
   when policy requires the selected interconnect. Explicit overrides take
   precedence and fail visibly if invalid.
6. Re-probe on link/address/route changes. A changed path can serve new
   connections only after peer authentication and route validation. Replacing
   a broken control connection reconciles its attempts as described below;
   it is not live stream migration. Removing a cable changes reachability,
   never membership or the accounting of unfinished work.

The resulting graph records **nodes, local shared-port groups, and verified
reachable endpoint pairs**. A same-subnet route or carrier cannot distinguish
a direct cable from a switch, reveal every unseen node, or establish an end-to-end
bandwidth guarantee. Mark cabling as owner-supplied or externally corroborated
when appropriate. On a larger graph, validate every needed conductor/worker
route; a physical ring is not permission to assume IP forwarding. M6a needs
TCP placement paths, not an RDMA fabric or a particular cable count.

### Layout classifier

Classify **physical-port groups** after collapsing the two Spark host paths
per port. Count distinct peer nodes after matching all observed addresses/MACs
to inventories obtained through enrolled TLS identity or already trusted
administrative SSH. Before that identity check, report candidate responders,
not a proven number of machines. Observed N includes the initiating node
exactly once plus distinct verified remote identities; discard self
advertisements and local-interface aliases from the peer count. ARP resolves IPv4 to a link-layer address;
matching its IP/MAC tuple to a peer's interface inventory supports a peer edge.
A transparent Ethernet switch forwards those same frames, so ARP does not
reveal whether the cable terminates at the peer or a switch. This follows
[ARP's address-mapping contract](https://www.rfc-editor.org/rfc/rfc826.html)
and [Ethernet bridge forwarding](https://docs.kernel.org/networking/bridge.html).

| Proposed layout | Automatic rule | Confidence and follow-up |
| --- | --- | --- |
| Single | Local physical inventory is complete and there are zero carrier-up QSFP port groups | Propose single-node setup. This describes current connectivity, not proof no other machines exist. Missing devices/driver/permissions yields unknown inventory. Existing enrolled peers becoming unreachable are degraded connectivity, never automatic single-node reconfiguration |
| Double, likely direct | One active QSFP port group on each of two identity-verified nodes; all responding peer paths for that group map to the other node, reciprocally | Propose two-node direct layout under the dedicated-network assumption. A switch with only two visible nodes is observationally equivalent; keep cabling unverified unless owner or independent evidence establishes it |
| Triple, likely direct | Three identity-verified nodes; each has two active QSFP port groups, each group maps only to one of the other nodes, and reciprocal inventories form all three edges of a triangle | Propose direct triangle. Two ports on just the initiating node are insufficient: they could form a chain or both lead to one switch. Confirm the peer-to-peer edge and preserve each port's path/subnet scope |
| N, switched/shared fabric | A single physical port group reaches multiple distinct verified peer nodes on its attached subnet(s); merge aliases from both host paths into one member set | Propose switched/shared-fabric layout, including 4+ nodes; also valid for 2/3-node switched deployments when known. Report observed N, not an exhaustive inventory of silent hosts. Verify all conductor routes; do not promise independent bandwidth per peer |
| Partial/other | Carrier but no responding verified peer, a truncated scan, conflicting IP/MAC claims, chain/ring, two ports to the same peer/fabric, asymmetric links, or missing peer inventory | Preserve the observed graph and list unresolved edges. Permit explicitly selected valid placement routes without forcing one of the canonical classes |

Choose a concrete direct-pair/triangle suggestion only when no contradictory
response or missing required reciprocal inventory remains. Mark the discovery
round complete or truncated separately from topology confidence; even a complete
round can miss a powered-off or filtered peer. Layout is setup/display metadata,
not membership, authority, forwarding configuration, or proof of model support.
Only the current two-node physical inventory has hardware evidence here; single,
triangle and switched-N detection are design requirements awaiting validation.

### Dedicated-QSFP subnet scan

The owner's dedicated-network assumption applies to detected physical QSFP
ports, not management Ethernet, Wi-Fi, containers, arbitrary routes or all
RFC1918 addresses. In an explicit setup/refresh window, scan the existing
**directly attached IPv4 prefixes** for each selected QSFP path. Run this even
when a neighbor hint finds one peer, since a switched subnet may contain more.
Never invent a peer IP from a MAC suffix or assume a /24 or the owner's subnets.
No IP address assignment is implicit; carrier with no usable addressing falls
back to scoped IPv6/mDNS, trusted inventory or an explicit network-setup step.

Use interface-bound ARP resolution plus at most one bounded reachability probe
per address; an ARP responder is retained as a candidate even if ICMP is
filtered. Probe one explicitly advertised/configured jitLLM or administrative
endpoint during candidate verification, not a range of service ports. If raw
ARP access is unavailable, use source/interface-bound ordinary socket probes
and the resulting OS neighbor data; report that reduced method, never install
privileges or assume nonresponders absent. Each probe must have an on-link route
through the selected interface with no gateway. Pin/check interface and source
throughout the scan; stop that scope on link/address/route changes. Proxy ARP,
a shared next-hop MAC or spoofed responses cannot substitute for peer identity.

The initial scan budget is **1024 (interface, target IPv4) pairs per setup
window**, at most **32 outstanding address probes**, **64 targets/s**, and
**1 s per address probe** with no retry in that round, within the existing
**60 s discovery window**. Each address probe has at most one ARP request and
one reachability request (OS ARP retransmissions must fit the probe/deadline
budget and be reported if the fallback cannot enforce that packet cap).
Separate authenticated candidate verification retains the four-probe/3 s
limits. Address-probe sockets also consume the node-wide connection and
transport-memory budgets; the 32-probe cap is not an additional unaccounted
pool. Scan records and raw/neighbor receive buffers are charged to setup,
with a 1 MiB node-wide scan-memory cap and a 2 KiB per-response parse limit;
over-budget input is dropped and the discovery round marked partial. These are policy bounds, not performance measurements.

Enumerate host addresses using the actual prefix semantics, excluding local
addresses and subnet network/broadcast addresses where applicable; /31 and
/32 are not treated as ordinary broadcast subnets. Deduplicate repeated
(interface, target) entries, but do not erase interface scope just because two
paths have overlapping prefixes. Round-robin scopes so one large subnet cannot
starve the other active ports. An aggregate range larger than 1024 pairs is
partially scanned and labelled truncated with unvisited ranges; explicit seeds
or a narrower operator-selected scope can finish discovery in another window.
A non-default narrowed scope must be a subset of an attached QSFP prefix.
IPv6 uses scoped neighbor/multicast discovery and explicit seeds, never a /64
sweep. Bound unauthenticated responder storage to 1024 address observations;
overflow, candidate-cap exhaustion or deadline expiration marks the result
partial. Compact to the existing 64 candidates/16 addresses per node only
when peer identity supports the merge; reaching a bound is not a smaller N.
Do not perform an automatic background retry or expand to a default route.

Schema v2 adds `network.subnet_scan = "dedicated-qsfp" | "off"`, defaulting to
`"dedicated-qsfp"` for setup when `network.discovery = "bootstrap"`.
`discovery = "off"` suppresses all discovery scans regardless of that setting.
An explicit interface override cannot cause automatic management-subnet scans;
only a detected QSFP physical group qualifies for this default. Setup can accept
an explicit narrower scan scope without storing it as cluster membership.
This revises D-038's earlier no-sweep rule; enrollment, TLS and session fencing
are unchanged. No scan was executed against the owner's network for this change.

## Evidence from the owner's two Sparks

Read-only probes on **2026-09-22**, over existing SSH from the x86-64 workstation,
found the same layout on both nodes. No sudo, service installation, network
writes, active bandwidth test or jitLLM discovery/enrollment execution occurred.

| Observation | Result on both nodes | Detection consequence |
| --- | --- | --- |
| Platform | DMI vendor `NVIDIA`, product `NVIDIA_DGX_Spark`, board `P4242` | Identifies the tested profile, not a universal GB10 layout |
| Adapter | `mlx5_core`, PCI vendor/device `15b3:1021`, subsystem `15b3:21ec`; firmware `28.45.4028 (NVD0000000087)` | Corroborates the profile independently of netdev naming |
| Physical grouping | Four netdevs share one nonempty local `phys_switch_id`; `phys_port_name` splits them into two `p0` and two `p1` entries. devlink agrees with physical port 0/1 | Groups two host paths per physical QSFP port; the switch ID differs between nodes and is not a remote-switch identity |
| Active right port | `0000:01:00.1` / `enp1s0f1np1` and `0002:01:00.1` / `enP2p1s0f1np1`: carrier 1, 200000 Mb/s, MTU 1500 | One active shared 200 Gb/s port per node, not 400 Gb/s |
| Inactive left port | Matching `.0` functions: carrier 0, speed -1 | Negative speed means unavailable, not usable bandwidth |
| Misleading field | `dev_port=0`, `dev_id=0x0` on all four interfaces | Neither field alone distinguishes these ports |
| IP/RDMA | Both active interfaces have IPv4 and link-local IPv6; corresponding RDMA ports are ACTIVE. Kernel routes select each peer's matching interconnect/source | Existing network is usable as discovery input without renumbering |
| Neighbors | Both peer interconnect addresses present but STALE | Candidate hints, not live probes or peer authentication |
| Sync plan | `/etc/netplan/99-nvidia-sync-cluster.yaml` not readable by the SSH user | Runtime must not need privileged access to it |

`spark` has `10.100.208.2` and `10.100.209.2`; `spark-b` has `.1` on those
same subnets. These are the owner's deployment, not defaults. Route lookups
in both directions selected the expected source/interface. The known direct
right-port cable is owner-supplied topology, corroborated by the previous
[interconnect baseline](experiments/interconnect/README.md); the new probes
did not establish remote identity through jitLLM or measure throughput.
Kernel was `7.0.0-1019-nvidia` on both nodes. Raw probe output stays external.

[NVIDIA's Spark port table](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html)
corroborates the two host paths per physical port and the left/right mapping.
[NVIDIA Sync's inspection guide](https://docs.nvidia.com/sync/latest/cluster-network-inspection.html)
distinguishes the configured network from workload orchestration. Both were
checked on 2026-09-22; jitLLM consumes the network it finds, not a claim that
Sync enrolled jitLLM members. No NVIDIA setup script was executed or copied.

## Configuration v2

Use **UTF-8 TOML**, `schema_version = 2`, with no include directives, environment expansion,
secret interpolation or executable hooks. D-063 lets the node-local document be split
into drop-in fragments (a key other than `schema_version` set in two files is fatal); the shared document
stays one file. This supersedes the unimplemented v1 draft with the explicit scan policy;
no older runtime format exists. Both shared and local documents use v2. Reject unknown versions/keys,
duplicate keys/IDs, invalid types/ranges and conflicting conductor declarations
before opening remote listeners or dispatching work. Configuration files are
limited to 1 MiB, 64 enrolled nodes and 16 seed endpoints per node in v2;
these are v2 safety ceilings, not hardcoded physical topology.
Larger deployments require a deliberate schema/limit review.

A cluster-wide document contains the public membership/trust policy. Each
node has a local document for identity, credential paths, listening interfaces
and resource limits. Setup writes them atomically with restrictive ownership;
private keys never appear inline or in diagnostics. The shared file is copied
byte-for-byte to all members: its SHA-256 plus revision is the membership
digest, so comments/whitespace changes require redistribution too. No implicit
merge of differing membership views is permitted.

| Shared field | Required meaning |
| --- | --- |
| `schema_version`, `cluster_id`, `revision` | Version 2, generated UUID, monotonically increasing positive integer for an explicit change |
| `conductor_id` | Exactly one node UUID present in `members`; no inferred/elected role |
| `members` | Array of node UUID, display label, SHA-256 certificate public-key pin, seed address array. IDs and pins unique; labels have no authority |
| `network.discovery` | `bootstrap` or `off`; bootstrap allows only setup windows, never ambient enrollment |
| `network.subnet_scan` | `dedicated-qsfp` (default) or `off`; bounded setup-only scans of attached QSFP prefixes, never management networks |
| `network.path_policy` | `auto` or `explicit`; auto uses the detection algorithm above for enrolled nodes |
| `network.allow_management_fallback` | Boolean, default false; setup offers enabling it if desired |

| Node-local field | Required meaning |
| --- | --- |
| `schema_version`, `cluster_file`, `node_id` | Version 2, absolute shared-file path, this enrolled UUID |
| `credentials.ca_file`, `certificate_file`, `private_key_file` | Absolute paths to cluster CA and this node's credentials; no downloads or key generation during normal runtime startup |
| `control.port` | Integer 1–65535, explicit TCP port, no fixed application-wide default; setup can propose an available port |
| `control.interfaces` | `auto` (validated selected interconnects) or explicit selector array. Selector uses adapter identity/physical port or an exact interface name override; auto excludes management unless fallback was enabled |
| `control.peer_scopes` | Optional table mapping member UUIDs to an observer-local interface selector for shared unscoped IPv6 link-local seeds |
| `client.bind` | Conductor only, default loopback; enabling remote binding requires TLS and client authentication configuration under D-014. Workers reject a client-listener setting |
| `limits.profile` | `initial-v2`, selecting all bounds below. *Revised by D-102 (2026-10-03), before M4 builds it:* the bounds become permissive defaults derived from each node's resources, and each one an owner may set (stricter included) as a node-local key, as `[client]`'s are (runtime-serving.md); peers agree on the protocol's own framing values, not on local pool sizes |

Seed addresses use `IPv4:port`, `[IPv6]:port` or `DNS-name:port` strings.
Shared IPv6 seeds never carry an observer-local zone/interface name. A node
expands a link-local seed over its selected local interfaces, or uses a
node-local `control.peer_scopes` binding from member UUID to local interface
selector; only authenticated route checks make a candidate usable. DNS results remain
untrusted until mutual TLS identity checks. Setup collects addresses from the
inventories; node IDs are generated and survive address changes. The immutable
membership file does not contain transient runtime incarnations, report
freshness, dynamic capacity or cached state hints. Address discovery may find
new endpoints for an existing identity; it cannot add a new identity.

Example shared skeleton (future schema; values illustrate structure, not the
owner's actual enrollment or credentials):

```toml
schema_version = 2
cluster_id = "af564a6b-8b4e-4528-8140-e50b92b40001"
revision = 1
conductor_id = "af564a6b-8b4e-4528-8140-e50b92b40002"

[network]
discovery = "bootstrap"
subnet_scan = "dedicated-qsfp"
path_policy = "auto"
allow_management_fallback = false

[[members]]
id = "af564a6b-8b4e-4528-8140-e50b92b40002"
label = "desk"
public_key_sha256 = "REPLACE_WITH_ENROLLED_KEY_HASH"
seeds = ["10.100.208.2:7443", "10.100.209.2:7443"]

[[members]]
id = "af564a6b-8b4e-4528-8140-e50b92b40003"
label = "second-node"
public_key_sha256 = "REPLACE_WITH_ENROLLED_KEY_HASH"
seeds = ["10.100.208.1:7443", "10.100.209.1:7443"]
```

Local selectors are strings `ifname:<exact-name>` or
`port:<local-phys_switch_id>/<phys_port_name>`. `control.interfaces` accepts
`"auto"` or a nonempty array of those strings; `control.peer_scopes` values
use the same selector syntax. `path_policy = "explicit"` requires a selector
array and a seed for each remote member. IDs are canonical lowercase UUIDs,
key pins are lowercase 64-digit SHA-256 hex of DER SubjectPublicKeyInfo,
labels are nonempty UTF-8 strings of at most 128 bytes, and paths must be
absolute. No leading-zero/negative/overflow acceptance for bounded integers.
Per-node identity/port/path policy mismatches reject startup or peer handshake.

Example local document for the first member (paths are illustrative until M1).
D-063 makes this document the node's `/etc/jitllm/jitllm.toml` plus its
`jitllm.d/` fragments, extended with `[storage]` and the other node keys; a standalone node omits `cluster_file`,
`node_id`, `[credentials]` and `[control]`:

```toml
schema_version = 2
cluster_file = "/etc/jitllm/cluster.toml"
node_id = "af564a6b-8b4e-4528-8140-e50b92b40002"

[credentials]
ca_file = "/etc/jitllm/credentials/ca.pem"
certificate_file = "/etc/jitllm/credentials/node.pem"
private_key_file = "/etc/jitllm/credentials/node-key.pem"

[control]
port = 7443
interfaces = "auto"

[client]
bind = "127.0.0.1:8114"

[limits]
profile = "initial-v2"
```

The placeholder hashes intentionally fail semantic validation. Port 7443 and
these seed addresses are examples, not assigned product ports or subnet rules.
A setup-generated file contains real hashes and observed/selected endpoints.
The syntax is settled here; D-063 sets the installed paths, and the parser
library and exhaustive diagnostic strings are selected in M1. Endpoint protocol/client settings
remain the separate named-client/API planning task; this schema does not
invent their authentication format or a remote-client default.

## Trust, startup and configuration changes

Enrollment requires an explicit setup action. Generate node keys on their
own nodes and exchange certificate requests through existing authenticated
administrative access, or import a prepared enrollment bundle locally. The
setup authority signs node certificates with a private cluster CA and writes
the approved node IDs/public-key pins into the shared document. Keep the CA
signing key out of serving workers. Bootstrap advertisements and unauthenticated
network traffic cannot obtain certificates or approve their own key pins.
An unknown SSH host key needs normal administrative trust establishment;
setup never disables host-key checks or copies private SSH keys.

Use TLS 1.3 mutual authentication for internal control and response transport.
Check certificate validity, the configured CA chain, enrolled public-key pin
and a node-ID subjectAltName URI `urn:jitllm:<cluster-uuid>:node:<node-uuid>`
with the configured cluster/node identity.
The certificate identity is the node, not a detected IP address. Authenticate
both ends before capability/state reports or inference data. Client-facing
credentials are separate. A secure direct QSFP connection has the same checks
as a management-network connection; physical proximity is not authentication.
No unauthenticated runtime status or model-list endpoint is exposed to discovery.

Normal startup verifies files, local node identity and the conductor's single
process lock. Every runtime gets a fresh random incarnation; every conductor
start durably increments a monotonic **authority epoch** before accepting
clients. Setup initializes epoch state once at enrollment; normal startup
never initializes a missing state file, and never runs a node standalone
while its enrollment or epoch state exists. D-063's fixed-path enrollment
anchor, `/var/lib/jitllm/enrollment`, keeps that true when a relocated
`state` role and its configuration are lost. Leaving a cluster is an
explicit setup step that retires that state and removes the anchor. Missing/corrupt authority state or
a rollback detected against worker epoch floors fails closed; it is not
reset to zero. Each worker atomically and durably records the highest accepted conductor
(epoch, incarnation) pair before acknowledging it and rejects lower epochs. The cluster ID, membership revision/digest,
conductor ID, both runtime incarnations and protocol version must match the
handshake before admission. Equal epochs may reconnect only for the same durably recorded
conductor incarnation; a newer process needs a larger epoch. A locally valid
whole-state rollback is not detectable without a non-rolled-back peer or
external authority. Restoring/cloning any authority-state backup therefore
requires operator fencing and fresh enrollment/epoch state coordinated with
all remaining workers; ordinary startup does not promise rollback detection
when every relevant durable copy was restored.

Replacing a connection creates a worker-issued fresh session nonce and fences
the old session at the worker's serialized admission boundary. Only one session
per worker/conductor authority admits commands, even with multiple valid network
paths. Session creation has fresh challenge/response and is bound to its live
TLS connection. The worker assigns an increasing handshake generation,
invalidates older pending challenges when designating a newer replacement,
and atomically installs only the currently designated generation. Thus a
delayed completion cannot supersede a session installed by a newer handshake.
The old session is fenced when replacement installs, not while an incomplete
handshake merely waits; admission-state changes serialize with that install.
Old-session commands, grants and stream chunks are rejected. Reconciliation
reports old attempts and starts cancellation of orphaned work; no new admission
uses unaccounted resources. Incomplete cleanup can coexist with new work only
when the live local ledger admits it safely under D-037/D-050.

One supervisor-held per-node process lock prevents a second runtime using the
same local identity. Authority epochs fence stale traffic, but are **not an
election algorithm** and do not make two cloned conductors safe. Moving the
conductor or changing membership/trust/CA requires an explicit coordinated
stop: stop admission and drain/cancel work, verify the old conductor stopped
or is isolated from every participant, distribute the new revision and
credentials, and restart with a higher authority epoch. An unreachable old
conductor/node must be fenced out of communication before the new membership
is activated. Copying an identity/state directory to another live host is not
supported. Epoch/session records use crash-safe atomic durable updates; storage
failure keeps the node unavailable. Local recovery still proves consumer
retirement before reclaiming memory, regardless of process or epoch changes.

V2 has no hot membership/trust reload, partial membership acceptance or automatic
conductor replacement. A member with a mismatching revision/digest stays
unavailable; a missing matching member need not prevent whole-model work on
other independently safe members. Address changes of an enrolled node
can be detected without changing membership, provided policy and authentication
still pass. There is no durable stream replay or transparent request resumption.

## Internal messages and bounded state

Initial internal transport is an ordered TCP/TLS stream with a four-byte
unsigned big-endian payload length and UTF-8 JSON objects, protocol version 1.
Every message has a type and correlation identity; application records are
bounded before allocation. Reject malformed lengths/UTF-8, duplicate or unknown
keys/types, nesting over 16, and incompatible versions; terminate that session
and account for/cancel its attempts. No raw process addresses, device pointers
or unvalidated filesystem paths cross this boundary. Schema tests and an exact
per-message field catalog are M6a implementation gates; they cannot silently
change these identity, framing or safety semantics.

Message families are hello/session, heartbeat/capability summary, request
begin/body/end, admission outcome, cancel/status, response chunk/end and
reconciliation/retirement acknowledgement. After authentication, handshake
and session establishment precede all other families. Full request payloads
are chunked under the input cap; incomplete input never starts execution and
expires under a bounded receive deadline. Worker scheduling reads validated
records, not arbitrary remote calls into the allocator. Response data uses
bounded credits so a blocked consumer cannot consume control/cancel capacity:
reserve control-queue space and interleave control messages between bounded
data records. Socket writes are asynchronous; an unresponsive connection is
closed at its deadline and worker orphan cancellation follows.

Within a session, attempt sequence numbers increase by one on each request
begin. The worker advances an accepted/rejected high-water mark atomically
with creating an attempt or permanently rejecting it. Repeated/lower sequence
numbers return retained status or `retired`; they never execute again. A gap
fails the session rather than allocating a sparse unbounded ledger. Each
attempt ID includes its original session and sequence: a new session cannot
resubmit it as new work. Cancellation of an unstarted attempt atomically makes
it permanently non-startable before issuing a no-start receipt. Only that
receipt permits conductor rerouting to another node; a generic cancellation
acknowledgement or `retired` response does not. Explicitly rejected begins
also have a durable-for-the-session no-start outcome; session fencing prevents
late messages after the session is retired.

Keep records for outstanding work, including orphan cleanup, until locally
safe retirement. Keep terminal outcomes only within the bounded cache; their
removal cannot erase the session high-water rejection fence. A new runtime
incarnation rejects old sessions even if their caches are gone. If unresolved
work reaches the record cap, reject new admission rather than dropping its
accounting. Reconciliation is bounded/chunked; a missing detailed record yields
unknown failure, never a claim that the request did not run. Rerouting creates
a fresh target attempt only after the old target is permanently fenced from
starting its original attempt. Client retries remain separate requests.

The defaults below are **initial policy choices, not measured safe throughput,
model-size or latency limits**. They bound coordination overhead and failure
waiting; D-050's [reservation policy](reservation-policy.md) separately defines
resource-progress guarantees. This first profile has no numeric override negotiation; peers require the
same profile identity, while local memory pressure may reduce actual admission
below every ceiling. Charge actual allocations to each node.

*Revised by D-102 (2026-10-03), to apply before M4 builds the request
path:* limits come from real resources, not time or abuse caps. Each pool
below (transport, input, stream, control) is a share of the node's memory,
each count of sockets or attempts what descriptors and memory allow, and
each is a configurable node-local key, permissive by default. Time limits
on work become inactivity limits (no bytes, no credit, no progress for N
seconds) or owner options: a request's input that keeps arriving, a
deferred admission that waits behind progressing work and a stream whose
reader is slow (it pauses on credits) are never cut off by a clock.
Detection stays: heartbeats, the handshake and reconciliation deadlines and
status resolution detect a dead peer, and a lost peer still starts orphan
cancellation. The protocol's own framing (a network message's 64 KiB, a
control record's) stays fixed. The table keeps the initial values for
reference; M4's design pass fixes the derived ones.

| Limit | Initial value and meaning |
| --- | --- |
| Connections | 64 node-wide sockets total including at most 8 pending TLS/hello handshakes, at most 2 pending per source address; one admission session and at most one inventory probe per authenticated peer. Reserve a slot/buffer allowance for established authority control |
| Transport memory | 32 MiB node-wide pool for application/TLS/accepted-socket buffering; at most 512 KiB charged per connection. Account actual OS socket buffers (including kernel adjustments), bound library allocations and cert-chain input; refuse new sockets/handshakes before exceeding either bound. Listener backlog capped at 128, with OS networking overhead separately budgeted |
| Dedicated-QSFP address scan | 1024 interface/IPv4 target pairs and address observations/window, 32 pending address probes, 64 targets/s, 1 s deadline each; report truncation. This sub-budget is separate from authenticated verification |
| Setup discovery | 60 s window; at most 64 candidates and 16 addresses/candidate; at most 4 candidate-verification probes in flight, each with 3 s deadline; no retry beyond window |
| Network message | 64 KiB including the JSON object; bounded chunks use base64 and count encoded bytes too |
| Request input | 8 MiB decoded total per request, shared with client admission limit; 30 s input-receive deadline. Oversize input fails explicitly before inference. *D-102:* the client admission limit follows memory (runtime-serving.md) and the deadline becomes an inactivity limit |
| Worker outstanding attempts | 32 per node including deferred/receiving/running/cleanup records; terminal cache 256 records plus one high-water scalar per live session |
| Input buffers | 64 MiB node-wide decoded input pool, reserved before request-body acceptance, including conductor forwarding/local inputs |
| Conductor outstanding attempts | 256 total; the 32-attempt per-node ceiling still applies. Includes unknown attempts until reconciled or their session is fenced and the node takes sole cleanup ownership |
| Deferred admission | 30 s without a grant, then no-start rejection/cancellation. Once started, generation is not capped by this queue deadline. *D-102:* no default wait while admitted work progresses; an owner option |
| Stream buffering | 1 MiB response data per attempt at each hop, within an 8 MiB node-wide stream pool; lack of credits pauses producers. No credit for 30 s cancels the stream. *D-102:* pools from memory; a reader without credit only pauses its producer unless an owner sets an inactivity limit |
| Control buffering | Separate 1 MiB node-wide bounded control pool; at most 64 KiB each record. If unavailable, stop accepting/reading additional work and preserve existing cancellation/heartbeat capacity |
| Heartbeats | Every 2 s; reports older than 6 s are ineligible for new placement; 10 s without authenticated progress marks connection lost and starts orphan cancellation |
| Status resolution | 10 s to resolve uncertain dispatch for the client; fail explicitly when exhausted. Worker cleanup may outlast it and remains charged |
| Handshake/reconcile | 10 s handshake deadline, 30 s reconciliation exchange deadline; no new admission until reconciliation completes or explicitly reports still-charged cleanup |

Time is measured using local monotonic clocks, not cross-node wall-clock
subtraction. Health clocks advance only on authenticated current-session
messages, never discovery advertisements. Queue and buffer limits are admission
limits; a timeout never proves GPU/network/I/O completion. The 64 MiB input pool bounds total accepted request bodies independently
of the attempt count; preflight the declared decoded length against remaining
pool capacity before accepting chunks, and reject mismatched/excess lengths.
Other resident state can further reduce what can actually be admitted.
The conductor enforces the same pool for forwarding and local execution,
not merely the count of requests. The total input limit is the same for local and
remote execution and must be reconciled with named-client tests before M5/M6a.

## Health, placement and state affinity

A node progresses through disconnected, authenticated, reconciling, ready or
unavailable states. Ready requires matching membership/protocol, completed
startup ownership checks, a fresh capability/budget summary and a usable
transport path. “Ready” is only placement eligibility, not a capacity grant.
Backend/driver/device compatibility, prepared-artifact identity and executable
plan support are probed/reported by that node; link speed cannot establish
model support. GPU or memory recovery failure makes it unavailable even if
heartbeats still arrive.

Filter eligible nodes by exact plan/artifact compatibility and then apply
D-037's placement preference: avoid paging when a feasible placement exists,
use already prepared instances and compatible retained-state affinity where
helpful, and ask the chosen node to admit its complete envelope. Reported free
bytes cannot overrule a competing local admission. The artifact need not be
resident, but it must be locally prepared/available before execution; M6a
does not silently transfer checkpoints, KV or extents between nodes. If no
node can make progress, defer within bounds, time-slice at safe boundaries,
or explicitly reject under D-050's reservation policy.

Shared-prefix and conversation-continuation hints remain independent under
D-031. Retention hints are scoped to the requesting security principal and
compatibility identity; validate them locally at use, without logging prompt
content. Stale affinity falls back to recomputation from the supplied history
or a declared error, never a wrong conversation. No cache hit refreshes an
unrelated continuation. A routed conversation can stay on an existing instance;
automatic replica creation remains outside M6a.

## Required validation and handoff

Implementation must exercise all layout-classifier rows, including direct pairs
versus a switch with two visible nodes, a true reciprocal triangle versus a
chain, four-plus switched nodes, dual-path address aliases, self advertisements
and initiating-node counting, silent peers,
proxy ARP, /31 addressing, overlapping prefixes, truncated/overflowing scans,
management-interface exclusion and route changes during a scan. It must also
exercise renamed interfaces, empty/inconsistent port IDs,
both QSFP ports active, one host path down, management fallback disabled,
IPv4-only/IPv6-scoped networks, stale neighbors, blocked multicast, a switched
network and ambiguous candidate peers. Check node deduplication across multiple
addresses and port-speed accounting without pretending two paths are two cables.
Unauthenticated discovery, an unexpected node certificate, conflicting cluster
revisions, cloned identities, and delayed old-session commands must not admit
work. Test invalid config/frame lengths, record-cap exhaustion, input/stream
backpressure, stale reports, lost acceptance, queued cancel/reroute, worker and
conductor restart, state-expiry races and unrecoverable consumer completion.
M8 separately validates RDMA/collective paths and multi-rank failure behavior.

Builder handoff, 2026-09-22: extended the existing uncommitted D-037 planning
work; collected read-only sysfs/devlink/RDMA/IP/route evidence on both Sparks;
checked current NVIDIA primary documentation; recorded observation versus
policy distinctions. TOML syntax/local documentation links and whitespace
were checked on the workstation. No jitLLM parser, enrollment, listener,
transport or runtime tests exist yet for this design; none were claimed run.
No network settings changed. Raw samples and temporary probes stay outside
Git. This is a design completion, with implementation gates still outstanding.

Independent review, 2026-09-22: reviewed the complete uncommitted D-037/D-038
planning change against the handoffs, pager invariants and M0 scope; no
substantive defects found. Checked both external Spark inventory captures and
the linked NVIDIA primary sources against the stated hardware observations,
and re-ran the whitespace check on the x86-64 workstation. Reviewed discovery
versus enrollment, observer-local path scopes, bounded transport/input state,
epoch/session fencing, uncertain-attempt handling and retained-state affinity.
This was a documentation/evidence review, not execution of the proposed
parser, discovery, authentication or runtime protocols; implementation and
hardware challenge gates remain outstanding.

A separate adversarial pass found and prompted fixes for observer-local IPv6
scopes, pre-authentication connection/memory bounds, durable epoch/incarnation
state and its rollback limits, and concurrent handshake installation ordering.
The subsequent adversarial recheck was clean. These were design challenges;
no fault-injection or runtime tests were executed.

D-039 builder follow-up (2026-09-22): added canonical layout rules and the
owner-requested dedicated-QSFP subnet scan; revised the experimental shared/
local schema and limit profile to v2, retaining internal wire protocol v1.
Checked the ARP/bridge inference against primary specifications, parsed both
TOML examples and checked local links/whitespace on the workstation. No live
subnet scan, new physical layout test or runtime implementation was performed.

D-039 independent follow-up review (2026-09-22): checked the full uncommitted
result and the added classifier/scan contract against the prior review; no
substantive defects found. Reviewed shared-port/identity deduplication,
reciprocal pair/triangle evidence, switched-network ambiguity, partial scans,
management exclusion, route changes and bounded probe/response accounting.
Rechecked the linked ARP/bridge primary sources, independently parsed both
TOML examples with schema v2 and the expected scan/profile fields, and passed
the whitespace check on the workstation. The separate topology challenge
reported a clean result after clarifying that observed N includes the
initiating node. These are design reviews; no live scan or proposed runtime
protocol was executed, and the stated implementation gates remain owed.
