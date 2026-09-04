/// Metadata model for a fragment stored in human memory.
///
/// [Architecture Note - Human Storage Model]:
/// In accordance with the Human Server architecture, human nodes never persist fragment
/// plaintext to device storage. The human brain is the long-term storage location.
/// This model records ONLY metadata required to track custody, verification hashes,
/// sequence numbers, and expiration timestamps.
class StoredNodeFragment {
  final String fragmentId;
  final String memoryId;
  final int sequenceNumber;
  final int sizeBytes;
  final DateTime receivedAt;
  final DateTime expiresAt;
  final String assignmentId;
  final String verificationHash;
  final String status;

  const StoredNodeFragment({
    required this.fragmentId,
    required this.memoryId,
    required this.sequenceNumber,
    required this.sizeBytes,
    required this.receivedAt,
    required this.expiresAt,
    required this.assignmentId,
    required this.verificationHash,
    this.status = 'MEMORIZED',
  });

  bool get isExpired => DateTime.now().isAfter(expiresAt);

  factory StoredNodeFragment.fromMap(Map<String, dynamic> map) {
    return StoredNodeFragment(
      fragmentId: map['fragment_id'] as String? ?? '',
      memoryId: map['memory_id'] as String? ?? '',
      sequenceNumber: (map['sequence_number'] as num?)?.toInt() ?? 1,
      sizeBytes: (map['size_bytes'] as num?)?.toInt() ?? 0,
      receivedAt: map['received_at'] != null
          ? DateTime.tryParse(map['received_at'].toString()) ?? DateTime.now()
          : DateTime.now(),
      expiresAt: map['expires_at'] != null
          ? DateTime.tryParse(map['expires_at'].toString()) ?? DateTime.now()
          : DateTime.now(),
      assignmentId: map['assignment_id'] as String? ?? '',
      verificationHash: map['verification_hash'] as String? ?? '',
      status: map['status'] as String? ?? 'MEMORIZED',
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'fragment_id': fragmentId,
      'memory_id': memoryId,
      'sequence_number': sequenceNumber,
      'size_bytes': sizeBytes,
      'received_at': receivedAt.toIso8601String(),
      'expires_at': expiresAt.toIso8601String(),
      'assignment_id': assignmentId,
      'verification_hash': verificationHash,
      'status': status,
    };
  }
}
