<img width="1280" height="640" alt="git (1)" src="https://github.com/user-attachments/assets/8920b256-2ba8-4988-b824-5351134eb4bd" />

# Human Server 🧠

## Basic Details

### Team Name: Serverless

### Team Members

* Team Lead: Sivnand V - SCMS School of Engineering and Technology

### Project Description

Human Server is a decentralized storage system where humans act as temporary memory nodes. User information is never stored as permanent plaintext on centralized cloud servers. Instead, a message is fragmented on the sender device into randomized, overlapping word/phrase packets, distributed across multiple distinct human nodes for memorization, and later retrieved through coordinated human recall, where the memory owner reconstructs the exact original message using consensus voting.

### The Problem (that doesn't exist)

Cloud storage has become far too reliable. Modern data centers guarantee 99.999999999% durability, replicate data across continents in milliseconds, and remember everything indefinitely. Human Server solves the completely unnecessary problem of finding an inherently volatile storage medium that can forget, misplace details, go offline, sleep, or simply refuse to cooperate: **the human brain**.

### The Solution (that nobody asked for)

Human Server completely eliminates conventional disk-backed server storage. It replaces silicon drives with actual people.

```
Original Plaintext (Entered by Sender)
       │
       ▼ (Immediately wiped from client UI & memory)
Client-Side Overlapping Packets (Randomized phrase combinations)
       │
       ▼ (Routed through temporary ephemeral inbox)
Different Human Nodes (1 distinct human per packet, max 5 capacity)
       │
       ▼ (Human memorizes text, server drops delivery buffer)
Human Memorization (Biological retention only; metadata-only locally)
       │
       ▼ (Later, owner transmits recall telegraph)
Human Recall (Nodes manually type remembered text)
       │
       ▼ (Ephemeral transport buffer)
Consensus Reconstruction (Owner device reconstructs exact original message)
       │
       ▼ (Purges server transport rows)
Owner Recovery History (Retained locally on owner device only)
```

---

## Technical Details

### Technologies/Components Used

For Software:

* **Dart 3.7 & Flutter 3.29**: Cross-platform client framework targeting Windows Desktop & Android.
* **Supabase & PostgreSQL**: Metadata coordination, Row-Level Security (RLS), and ephemeral transport buffers.
* **PostgreSQL RPCs**: Server-enforced capacity limits, assignment routing, and retrieval lifecycle enforcement.
* **Client-Side SHA-256 (crypto package)**: Checksum verification during transport.
* **SharedPreferences**: User-scoped metadata and owner-device local recovery history.
* **Google Fonts**: Retro-modern telegraph aesthetic (Playfair Display, Space Mono, Courier Prime, Inter).
* **Git & GitHub Actions**: Automated multi-platform release packaging.

For Hardware:

* Android smartphone / Windows development computer
* Internet connection (HTTPS / WebSocket)
* No specialised electronic hardware required

---

## Core Architecture & How It Works

### 1. Zero Server Plaintext & Privacy Model
**Human Server does NOT store original plaintext permanently on the server.**
- When a user enters a secret or message (e.g. `THE MEETING IS AT 4:30 PM IN ROOM 204`) and presses Store/Dispatch, the client fragments the text into overlapping packets.
- The input field and in-memory plaintext are cleared immediately. The sender UI never displays the original message again.
- The server only holds ephemeral transport records (`fragment_delivery_inbox` and `fragment_recall_responses`), which are automatically deleted as soon as receipt or retrieval is completed.

### 2. Overlapping Packet Redundancy Model
Redundancy does NOT come from storing identical duplicate copies of the same fragment. Instead, Human Server utilizes an intentional phrase overlap model:
- **Packet A**: `THE MEETING + 4:30`
- **Packet B**: `4:30 + ROOM 204`
- **Packet C**: `ROOM 204 + THE MEETING`
- **Packet D**: `THE MEETING + ROOM 204`
- **Packet E**: `4:30 + THE MEETING`

Each packet represents a partial, non-complete observation of the original sentence paired with canonical slot indices.

### 3. Strict 1 Packet Per Human & 5-Slot Capacity
- **1 Packet Per Human**: An engine-level unique database index (`uq_assignment_memory_node_active`) physically prevents any single human node from receiving more than one packet from the same memory.
- **5-Slot Node Capacity**: Each human station is strictly capped at hosting **5 active memorized packets** at any time. Once a node recalls a fragment, its slot is retired and immediately freed for subsequent hosting.

### 4. Consensus Reconstruction
Because human memory is biological, humans might make minor typos or recall partial phrases. The owner client evaluates multiple overlapping observations across canonical slots and performs consensus voting:
- Eliminates duplicate words and redundant tokens.
- Discards isolated typo outliers.
- Deterministically reconstructs the original sentence regardless of arrival order.

### 5. Recovery History vs. Node Storage
It is critical to distinguish between node storage and recovery history:
* **Stored Node Fragments**: Held by human nodes acting as storage. Contains **metadata only** (sequence number, verification hash, expiration, status). **Zero plaintext** is ever stored on disk for node data. Counts toward the 5-slot limit.
* **Recovered Memory History**: Belongs to the **owner** who initiated and recovered their own memory. Contains the reconstructed plaintext locally on the owner's device for their personal log. Does **not** count toward node storage, is never shared with peers, and never appears in the station vault.
* **Multi-User Isolation**: Both node metadata and recovery history are strictly partitioned by authenticated Supabase UUID. Switching accounts on the same device guarantees zero cross-account data leakage.

---

## Implementation & Installation

### Direct Downloads (GitHub Releases)

Download pre-built release artifacts directly from the repository's [GitHub Releases](https://github.com/sivuiii/useless_project_siv/releases):

#### 📱 Android APK Installation
1. Download **`HumanServer-Android-v1.0.0.apk`** from [Releases](https://github.com/sivuiii/useless_project_siv/releases).
2. On your Android device, tap the downloaded APK to install.
3. If prompted by Android, allow *"Install unknown apps"* for your browser or file manager (standard for direct APK distribution outside Google Play).
4. Open **Human Server**, create an account, and initialize your node!

#### 💻 Windows Desktop Installation
1. Download **`HumanServer-Windows-v1.0.0.zip`** from [Releases](https://github.com/sivuiii/useless_project_siv/releases).
2. Extract the ZIP folder to a directory of your choice.
3. Double-click **`human_server.exe`** to launch the standalone desktop application.
4. If Windows SmartScreen displays a warning, click *"More info"* → *"Run anyway"* (expected for self-packaged open-source MVP binaries).

---

### Building from Source

#### Prerequisites
* Flutter SDK (version 3.29.x or higher, stable channel)
* Dart SDK (version 3.7.x or higher)
* Android SDK / Android Studio (for Android build)
* Visual Studio with Desktop development with C++ (for Windows build)

#### Clone & Install Dependencies
```bash
git clone https://github.com/sivuiii/useless_project_siv.git
cd useless_project_siv
flutter pub get
```

#### Run Tests & Static Analysis
```bash
flutter analyze
flutter test
```

#### Build Windows Desktop Executable
```bash
flutter build windows --release
```
*Output directory*: `build/windows/x64/runner/Release/` (contains `human_server.exe` and required runtime DLLs).

#### Build Android Release APK
```bash
flutter build apk --release
```
*Output file*: `build/app/outputs/flutter-apk/app-release.apk`.

---

## Project Documentation

### Screenshots

![Screenshot1](https://github.com/user-attachments/assets/8920b256-2ba8-4988-b824-5351134eb4bd)
*Human Server station dashboard showing node capacity slots (0/5 used), station status, and network synchronization*

![Screenshot2](https://github.com/user-attachments/assets/8920b256-2ba8-4988-b824-5351134eb4bd)
*Store screen demonstrating client-side fragmentation into overlapping packets and immediate plaintext clearance*

![Screenshot3](https://github.com/user-attachments/assets/8920b256-2ba8-4988-b824-5351134eb4bd)
*Retrieve screen showing Active Recoverable Memories vs Recovery History and real-time consensus reconstruction*

### Workflow Diagram

```
+-------------------------------------------------------------------------------+
|                             HUMAN SERVER WORKFLOW                             |
+-------------------------------------------------------------------------------+
|                                                                               |
|  [SENDER DEVICE]                                                              |
|   Enter Text -> Client-Side Overlapping Fragments -> Immediate Plaintext Wipe |
|         |                                                                     |
|         v                                                                     |
|  [DATABASE / EPHEMERAL TRANSPORT]                                             |
|   1 Packet per distinct human node (max 5 capacity per node)                  |
|         |                                                                     |
|         v                                                                     |
|  [RECIPIENT NODES]                                                            |
|   Human reads fragment -> Confirms MEMORIZED -> Ephemeral inbox row purged    |
|   (Local station saves METADATA ONLY, zero plaintext on disk)                 |
|         |                                                                     |
|         v                                                                     |
|  [OWNER RECOVERY REQUEST]                                                     |
|   Owner clicks REQUEST RECOVERY -> Recall prompt transmitted to human nodes   |
|         |                                                                     |
|         v                                                                     |
|  [HUMAN RECALL & CONSENSUS]                                                   |
|   Nodes type remembered text -> Owner client receives responses               |
|   Consensus voting resolves overlaps & typos -> Reconstructs original message |
|         |                                                                     |
|         v                                                                     |
|  [LIFECYCLE COMPLETION]                                                       |
|   Server purges recall transport buffer -> Memory status marked 'recovered'   |
|   Reconstructed plaintext saved into Owner Local Recovery History only        |
+-------------------------------------------------------------------------------+
```

For Hardware:

# Schematic & Circuit

![Circuit](https://github.com/user-attachments/assets/8920b256-2ba8-4988-b824-5351134eb4bd)
*No dedicated electronic circuit is required; the application operates using standard mobile smartphones and desktop PCs connected via HTTPS.*

![Schematic](https://github.com/user-attachments/assets/8920b256-2ba8-4988-b824-5351134eb4bd)
*System architecture consists of user client nodes connected to the PostgreSQL/Supabase coordination backend over secure endpoints.*

# Build Photos

![Components](https://github.com/user-attachments/assets/8920b256-2ba8-4988-b824-5351134eb4bd)
*Android smartphone, development computer, and network connection used to build, test, and run Human Server.*

![Build](https://github.com/user-attachments/assets/8920b256-2ba8-4988-b824-5351134eb4bd)
*Testing multi-node distribution and consensus reconstruction across 8 authenticated test accounts.*

![Final](https://github.com/user-attachments/assets/8920b256-2ba8-4988-b824-5351134eb4bd)
*Final Human Server release running on Android and Windows.*

---

### Project Demo

# Video

[Demo Video Link: To be attached with submission]
*The demo video demonstrates the complete Human Server workflow: entering a message, instant sender plaintext clearance, distributing overlapping packets to distinct human nodes, human memorization, capacity slot tracking, requesting recovery, collecting recall responses, consensus reconstruction, and local owner history preservation.*

# Additional Demos

* GitHub Releases: [Releases](https://github.com/sivuiii/useless_project_siv/releases)

---

## Project Status & MVP Limitations

* **Status**: **MVP Release Candidate (v1.0.0+1)**. The end-to-end flow has been verified across 8 real test accounts.
* **Cognitive Capacity**: Nodes are capped at 5 memorized fragments to maintain humane cognitive load.
* **Network Synchrony**: Node communication operates via polling intervals.
* **Human Volatility**: If insufficient human nodes recall their entrusted fragments, reconstruction may be partial or incomplete. This is by design: humans are the storage.

---

## Team Contributions

Solo Project by **Sivnand V**.

---

Made with 🛠️ at TinkerHub Useless Projects

![Static Badge](https://img.shields.io/badge/TinkerHub-24?color=%23000000&link=https%3A%2F%2Fwww.tinkerhub.org%2F)
![Static Badge](https://img.shields.io/badge/UselessProjects--26-26?link=https%3A%2F%2Ftinkerhub.org%2Fevents%2F1M8ORET9A1%2Fuseless-projects-3.0)
