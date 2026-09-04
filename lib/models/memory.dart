class MemoryModel {
  final String id;
  final String ownerUserId;
  final DateTime createdAt;
  final DateTime expiresAt;
  final int fragmentCount;
  final String status;

  const MemoryModel({
    required this.id,
    required this.ownerUserId,
    required this.createdAt,
    required this.expiresAt,
    required this.fragmentCount,
    required this.status,
  });

  bool get isActive => status == 'active';

  factory MemoryModel.fromMap(Map<String, dynamic> map) {
    return MemoryModel(
      id: map['id'] as String? ?? '',
      ownerUserId: map['owner_user_id'] as String? ?? '',
      createdAt: map['created_at'] != null
          ? DateTime.tryParse(map['created_at'].toString()) ?? DateTime.now()
          : DateTime.now(),
      expiresAt: map['expires_at'] != null
          ? DateTime.tryParse(map['expires_at'].toString()) ?? DateTime.now()
          : DateTime.now(),
      fragmentCount: (map['fragment_count'] as num?)?.toInt() ?? 1,
      status: map['status'] as String? ?? 'active',
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'owner_user_id': ownerUserId,
      'created_at': createdAt.toIso8601String(),
      'expires_at': expiresAt.toIso8601String(),
      'fragment_count': fragmentCount,
      'status': status,
    };
  }
}
