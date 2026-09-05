/// Represents a successfully recovered memory in the OWNER device's local recovery history.
/// 
/// [Architecture Invariants]:
/// - Belongs strictly to the OWNER who requested and reconstructed the memory.
/// - May contain reconstructed plaintext locally on the owner's device.
/// - Does NOT count toward the 5-node-memory-slot limit.
/// - Is NOT a StoredNodeFragment.
/// - Must NEVER be used as a node fragment for distribution.
/// - Must NEVER appear in Station / Biological Storage Vault.
class RecoveredMemory {
  final String memoryId;
  final String reconstructedPlaintext;
  final DateTime recoveredAt;
  final int packetCount;

  const RecoveredMemory({
    required this.memoryId,
    required this.reconstructedPlaintext,
    required this.recoveredAt,
    this.packetCount = 0,
  });

  String get shortId =>
      memoryId.length >= 8 ? memoryId.substring(0, 8).toUpperCase() : memoryId;

  Map<String, dynamic> toMap() => {
        'memory_id': memoryId,
        'reconstructed_plaintext': reconstructedPlaintext,
        'recovered_at': recoveredAt.toIso8601String(),
        'packet_count': packetCount,
      };

  factory RecoveredMemory.fromMap(Map<String, dynamic> map) => RecoveredMemory(
        memoryId: map['memory_id'] as String? ?? '',
        reconstructedPlaintext: map['reconstructed_plaintext'] as String? ?? '',
        recoveredAt: map['recovered_at'] != null
            ? DateTime.tryParse(map['recovered_at'].toString()) ?? DateTime.now()
            : DateTime.now(),
        packetCount: (map['packet_count'] as num?)?.toInt() ?? 0,
      );
}
