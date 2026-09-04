class UserStoredMemory {
  final String id;
  final DateTime createdAt;
  final DateTime expiresAt;
  final int fragmentCount;
  final String status;

  const UserStoredMemory({
    required this.id,
    required this.createdAt,
    required this.expiresAt,
    required this.fragmentCount,
    required this.status,
  });

  factory UserStoredMemory.fromMap(Map<String, dynamic> map) {
    return UserStoredMemory(
      id: map['id'] as String? ?? '',
      createdAt: map['created_at'] != null
          ? DateTime.tryParse(map['created_at'].toString()) ?? DateTime.now()
          : DateTime.now(),
      expiresAt: map['expires_at'] != null
          ? DateTime.tryParse(map['expires_at'].toString()) ?? DateTime.now()
          : DateTime.now(),
      fragmentCount: (map['fragment_count'] as num?)?.toInt() ?? 0,
      status: map['status'] as String? ?? 'active',
    );
  }

  String get shortId => id.length >= 8 ? id.substring(0, 8).toUpperCase() : id;
  String get memoryId => id;
}

