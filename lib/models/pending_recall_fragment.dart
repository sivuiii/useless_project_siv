class PendingRecallFragment {
  final String retrievalId;
  final String assignmentId;
  final String fragmentId;
  final String memoryId;
  final int sequenceNumber;
  final int sizeBytes;
  final String expectedHash;
  final DateTime retrievalCreatedAt;
  final String senderUsername;
  final List<int> canonicalIndices;

  const PendingRecallFragment({
    required this.retrievalId,
    required this.assignmentId,
    required this.fragmentId,
    required this.memoryId,
    required this.sequenceNumber,
    required this.sizeBytes,
    required this.expectedHash,
    required this.retrievalCreatedAt,
    this.senderUsername = 'UNKNOWN',
    this.canonicalIndices = const [],
  });

  factory PendingRecallFragment.fromMap(Map<String, dynamic> map) {
    return PendingRecallFragment(
      retrievalId: map['retrieval_id'] as String? ?? '',
      assignmentId: map['assignment_id'] as String? ?? '',
      fragmentId: map['fragment_id'] as String? ?? '',
      memoryId: map['memory_id'] as String? ?? '',
      sequenceNumber: (map['sequence_number'] as num?)?.toInt() ?? 0,
      sizeBytes: (map['size_bytes'] as num?)?.toInt() ?? 0,
      expectedHash: map['expected_hash'] as String? ?? '',
      retrievalCreatedAt: map['retrieval_created_at'] != null
          ? DateTime.tryParse(map['retrieval_created_at'].toString()) ?? DateTime.now()
          : DateTime.now(),
      senderUsername: (map['sender_username'] as String?)?.trim().isNotEmpty == true
          ? map['sender_username'] as String
          : 'UNKNOWN',
      canonicalIndices: (map['canonical_indices'] as List<dynamic>?)
              ?.map((e) => (e as num).toInt())
              .toList() ??
          const [],
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'retrieval_id': retrievalId,
      'assignment_id': assignmentId,
      'fragment_id': fragmentId,
      'memory_id': memoryId,
      'sequence_number': sequenceNumber,
      'size_bytes': sizeBytes,
      'expected_hash': expectedHash,
      'retrieval_created_at': retrievalCreatedAt.toIso8601String(),
      'sender_username': senderUsername,
      'canonical_indices': canonicalIndices,
    };
  }
}
