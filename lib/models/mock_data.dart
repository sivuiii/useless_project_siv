class UserProfileMock {
  final String username;
  final int credits;
  final double reliability;
  final int activeMemoriesCount;
  final int maxMemoriesLimit;
  final bool isNodeOnline;
  final String nodeId;
  final String joinDate;

  const UserProfileMock({
    required this.username,
    required this.credits,
    required this.reliability,
    required this.activeMemoriesCount,
    required this.maxMemoriesLimit,
    required this.isNodeOnline,
    required this.nodeId,
    required this.joinDate,
  });
}

class ActivityItemMock {
  final String title;
  final String timestamp;
  final String type; // 'store', 'retrieve', 'hosted_verify', 'credit_earned'
  final String detail;

  const ActivityItemMock({
    required this.title,
    required this.timestamp,
    required this.type,
    required this.detail,
  });
}

class HostedFragmentMock {
  final String fragmentId;
  final String payloadHash;
  final String size;
  final String expiresAt;
  final String lastVerified;
  final int totalVerifications;

  const HostedFragmentMock({
    required this.fragmentId,
    required this.payloadHash,
    required this.size,
    required this.expiresAt,
    required this.lastVerified,
    required this.totalVerifications,
  });
}

class RecalledMemoryMock {
  final String promptMatch;
  final String content;
  final String storedDate;
  final String expiresDate;
  final double confidenceScore;
  final int respondingNodesCount;

  const RecalledMemoryMock({
    required this.promptMatch,
    required this.content,
    required this.storedDate,
    required this.expiresDate,
    required this.confidenceScore,
    required this.respondingNodesCount,
  });
}

class MockDataRepository {
  static const currentUser = UserProfileMock(
    username: 'node_master_42',
    credits: 480,
    reliability: 98.4,
    activeMemoriesCount: 2,
    maxMemoriesLimit: 5,
    isNodeOnline: true,
    nodeId: 'HS-NODE-8839-AX',
    joinDate: 'Jan 2026',
  );

  static final List<ActivityItemMock> recentActivities = [
    const ActivityItemMock(
      title: 'Memory Stored',
      timestamp: '10 mins ago',
      type: 'store',
      detail: 'Distributed chunk payload to 3 nodes (-50 Credits)',
    ),
    const ActivityItemMock(
      title: 'Recall Verification Served',
      timestamp: '1 hour ago',
      type: 'hosted_verify',
      detail: 'Responded to recall challenge #4921 (+15 Credits)',
    ),
    const ActivityItemMock(
      title: 'Memory Retrieved',
      timestamp: 'Yesterday',
      type: 'retrieve',
      detail: 'Recalled "Cabin Wi-Fi key & alarm code" (99.8% match)',
    ),
    const ActivityItemMock(
      title: 'Node Heartbeat Ping',
      timestamp: '2 hours ago',
      type: 'credit_earned',
      detail: 'Completed hourly node uptime check (+5 Credits)',
    ),
  ];

  static final List<HostedFragmentMock> assignedFragments = [
    const HostedFragmentMock(
      fragmentId: 'FRAG-9921-A',
      payloadHash: 'e3b0c44298fc1c149afbf4c8996fb924',
      size: '1.2 KB',
      expiresAt: '2026-08-15 (5 mos remaining)',
      lastVerified: '12 mins ago',
      totalVerifications: 142,
    ),
    const HostedFragmentMock(
      fragmentId: 'FRAG-4418-C',
      payloadHash: '8f4e0012a59a721d604b39b0a7019808',
      size: '0.8 KB',
      expiresAt: '2026-07-20 (4 mos remaining)',
      lastVerified: '2 hours ago',
      totalVerifications: 89,
    ),
    const HostedFragmentMock(
      fragmentId: 'FRAG-1044-F',
      payloadHash: '4a1b023910c2838421098efc77112001',
      size: '2.4 KB',
      expiresAt: '2026-09-01 (6 mos remaining)',
      lastVerified: '1 day ago',
      totalVerifications: 34,
    ),
  ];

  static final List<RecalledMemoryMock> mockRecalledMemories = [
    const RecalledMemoryMock(
      promptMatch: 'Cabin Wi-Fi & Gate Alarm Code',
      content: 'Wi-Fi Name: PineWood_5G\nPassword: SummerCabin2026!\nGate Alarm Code: 4921#',
      storedDate: '2026-02-14',
      expiresDate: '2026-08-14',
      confidenceScore: 0.998,
      respondingNodesCount: 5,
    ),
    const RecalledMemoryMock(
      promptMatch: 'Vintage Guitar Serial Number',
      content: '1974 Fender Stratocaster Serial: #583921. Insured under policy #POL-99201.',
      storedDate: '2026-01-10',
      expiresDate: '2026-07-10',
      confidenceScore: 0.962,
      respondingNodesCount: 4,
    ),
  ];
}
