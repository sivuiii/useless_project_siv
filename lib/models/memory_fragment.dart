class MemoryFragmentModel {
  final String id;
  final String memoryId;
  final int sequenceNumber;
  final int sizeBytes;
  final String hash;
  final DateTime createdAt;
  final DateTime expiresAt;

  const MemoryFragmentModel({
    required this.id,
    required this.memoryId,
    required this.sequenceNumber,
    required this.sizeBytes,
    required this.hash,
    required this.createdAt,
    required this.expiresAt,
  });

  factory MemoryFragmentModel.fromMap(Map<String, dynamic> map) {
    return MemoryFragmentModel(
      id: map['id'] as String? ?? '',
      memoryId: map['memory_id'] as String? ?? '',
      sequenceNumber: (map['sequence_number'] as num?)?.toInt() ?? 1,
      sizeBytes: (map['size_bytes'] as num?)?.toInt() ?? 0,
      hash: map['hash'] as String? ?? '',
      createdAt: map['created_at'] != null
          ? DateTime.tryParse(map['created_at'].toString()) ?? DateTime.now()
          : DateTime.now(),
      expiresAt: map['expires_at'] != null
          ? DateTime.tryParse(map['expires_at'].toString()) ?? DateTime.now()
          : DateTime.now(),
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'memory_id': memoryId,
      'sequence_number': sequenceNumber,
      'size_bytes': sizeBytes,
      'hash': hash,
      'created_at': createdAt.toIso8601String(),
      'expires_at': expiresAt.toIso8601String(),
    };
  }
}
