# Human Server v1.0.0 — MVP

Welcome to the initial MVP release of **Human Server**, the biological distributed memory network where humans act as decentralized storage nodes!

---

## What is Human Server?

Human Server is an experiment in decentralized biological storage. Instead of storing original plaintext messages on centralized database disks or server filesystems, user messages are fragmented client-side into randomized, overlapping word/phrase packets and entrusted to distinct human nodes across the network.

When the memory owner later requests recovery, online human nodes are prompted to recall their memorized fragments. The owner client collects the responses and reconstructs the exact original message via canonical index consensus voting.

---

## Key Features in v1.0.0

* **Zero Server Plaintext Storage**: The original plaintext is fragmented on the client and never permanently stored on Supabase or database tables. Plaintext only exists in temporary transport buffers during memorization delivery and recall transmission.
* **Overlapping Packet Distribution**: Messages are split into randomized, overlapping word/phrase packets with redundancy built into the overlapping phrases rather than duplicate raw copies.
* **1 Packet Per Human Rule**: Database constraints and server RPCs physically guarantee that no single human node can receive more than one packet from the same memory.
* **Human Node Station Capacity**: Human nodes are strictly capped at hosting **5 active memorized packets** at any time. When a fragment is recalled, its slot is retired and immediately freed.
* **Transparent Attribution**: Recipient nodes see who entrusted them with a fragment (`MEMORY FROM: [USERNAME]`), establishing biological accountability.
* **Human Recall & Consensus Reconstruction**: Nodes type what they remember. The owner device absorbs minor human typos, redundant overlaps, and out-of-order responses using consensus voting to reconstruct the exact original sentence.
* **Owner-Scoped Recovery History**: Once successfully reconstructed, the recovered message is retained locally on the owner's device as a persistent recovery log. It never counts toward node storage, is never shared with peers, and survives logout and app restarts.
* **Multi-User Isolation**: All local station metadata and recovery history are strictly scoped to the active authenticated Supabase user UUID. Multiple accounts on the same device never see or overwrite each other's data.
* **Cross-Platform Releases**: Available as a standalone Windows desktop portable application and an Android APK.

---

## Downloadable Artifacts

| Platform | File | Description |
| :--- | :--- | :--- |
| **Android** | `HumanServer-Android-v1.0.0.apk` | Direct installable APK for Android devices |
| **Windows** | `HumanServer-Windows-v1.0.0.zip` | Standalone portable executable (extract and run) |

---

## Honest MVP Limitations

As an MVP release candidate, please note the following design characteristics:
1. **Human Dependency**: Retrieval depends on human nodes being online, remembering their assigned fragments, and responding to recall requests. If too many nodes fail to recall or remain offline indefinitely, consensus reconstruction may be incomplete.
2. **Network Polling**: Node synchronizations and recall polling currently run over secure HTTP RPC intervals (3s) rather than permanent background OS push notification services.
3. **Application Signing**: Binaries are self-contained release builds. Windows SmartScreen or Android's "Install Unknown Apps" prompt may appear upon first install.
4. **Node Capacity**: In this MVP, human storage capacity is capped at 5 active slots per node to respect human cognitive load.

---

## Installation Quick Start

* **Android**:
  1. Download `HumanServer-Android-v1.0.0.apk`.
  2. Allow installation from unknown sources if prompted.
  3. Launch Human Server, create an account, and start hosting or storing memories!
* **Windows**:
  1. Download and extract `HumanServer-Windows-v1.0.0.zip`.
  2. Double-click `human_server.exe` to launch.
  3. Sign in or create an account.
