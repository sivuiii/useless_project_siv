/// Ephemeral in-memory representation of a fragment delivery awaiting human memorization.
///
/// [Architecture Note]:
/// This class exists strictly in transient application memory while the memorization
/// modal is active. It is NEVER written to SharedPreferences, SQLite, files, or persistent storage.
/// Once the human confirms memorization, [clear] is called immediately to wipe the plaintext
/// and the backend deletes the transport buffer row.
class PendingDeliveryFragment {
  final String inboxId;
  final String assignmentId;
  final String fragmentId;
  final String memoryId;
  final int sequenceNumber;
  final int sizeBytes;
  final String expectedHash;
  final DateTime expiresAt;
  final DateTime? assignedAt;

  // Ephemeral plaintext - cleared upon memorization confirmation
  String _payloadText;
  String get payloadText => _payloadText;

  bool _isCleared = false;
  bool get isCleared => _isCleared;

  PendingDeliveryFragment({
    required this.inboxId,
    required this.assignmentId,
    required this.fragmentId,
    required this.memoryId,
    required this.sequenceNumber,
    required String payloadText,
    required this.sizeBytes,
    required this.expectedHash,
    required this.expiresAt,
    this.assignedAt,
  }) : _payloadText = payloadText;

  factory PendingDeliveryFragment.fromMap(Map<String, dynamic> map) {
    return PendingDeliveryFragment(
      inboxId: map['inbox_id'] as String? ?? '',
      assignmentId: map['assignment_id'] as String? ?? '',
      fragmentId: map['fragment_id'] as String? ?? '',
      memoryId: map['memory_id'] as String? ?? '',
      sequenceNumber: (map['sequence_number'] as num?)?.toInt() ?? 1,
      payloadText: map['payload_text'] as String? ?? '',
      sizeBytes: (map['size_bytes'] as num?)?.toInt() ?? 0,
      expectedHash: map['hash'] as String? ?? '',
      expiresAt: map['expires_at'] != null
          ? DateTime.tryParse(map['expires_at'].toString()) ??
              DateTime.now().add(const Duration(days: 180))
          : DateTime.now().add(const Duration(days: 180)),
      assignedAt: map['assigned_at'] != null
          ? DateTime.tryParse(map['assigned_at'].toString())
          : null,
    );
  }

  /// Immediately zeroes out the ephemeral plaintext from client memory
  void clear() {
    _payloadText = '';
    _isCleared = true;
  }
}
