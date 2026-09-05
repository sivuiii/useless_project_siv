class UserStoredMemory {
  final String id;
  final DateTime createdAt;
  final DateTime expiresAt;
  final int fragmentCount;
  final int packetCount;
  final int memorizedCount;
  final String status;
  final String retrievalStatus;
  final String? activeRetrievalId;

  const UserStoredMemory({
    required this.id,
    required this.createdAt,
    required this.expiresAt,
    required this.fragmentCount,
    this.packetCount = 0,
    this.memorizedCount = 0,
    required this.status,
    this.retrievalStatus = 'ready',
    this.activeRetrievalId,
  });

  bool get isUnderRecovery => retrievalStatus == 'under_recovery';
  bool get isReady => retrievalStatus == 'ready';
  bool get isCompleted =>
      retrievalStatus == 'completed' || status == 'recovered';

  factory UserStoredMemory.fromMap(Map<String, dynamic> map) {
    final frags = (map['packet_count'] as num?)?.toInt() ??
        (map['fragment_count'] as num?)?.toInt() ??
        0;
    final memorized = (map['memorized_count'] as num?)?.toInt() ?? 0;
    return UserStoredMemory(
      id: map['id'] as String? ?? '',
      createdAt: map['created_at'] != null
          ? DateTime.tryParse(map['created_at'].toString()) ?? DateTime.now()
          : DateTime.now(),
      expiresAt: map['expires_at'] != null
          ? DateTime.tryParse(map['expires_at'].toString()) ?? DateTime.now()
          : DateTime.now(),
      fragmentCount: frags,
      packetCount: frags,
      memorizedCount: memorized,
      status: map['status'] as String? ?? 'active',
      retrievalStatus: map['retrieval_status'] as String? ?? 'ready',
      activeRetrievalId: map['active_retrieval_id'] as String?,
    );
  }

  String get shortId => id.length >= 8 ? id.substring(0, 8).toUpperCase() : id;
  String get memoryId => id;
}

