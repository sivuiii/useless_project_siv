class UserProfile {
  final String id;
  final String username;
  final DateTime createdAt;
  final int credits;
  final double reliability;

  const UserProfile({
    required this.id,
    required this.username,
    required this.createdAt,
    required this.credits,
    required this.reliability,
  });

  factory UserProfile.fromMap(Map<String, dynamic> map) {
    return UserProfile(
      id: map['id'] as String? ?? '',
      username: map['username'] as String? ?? 'node_user',
      createdAt: map['created_at'] != null
          ? DateTime.tryParse(map['created_at'].toString()) ?? DateTime.now()
          : DateTime.now(),
      credits: (map['credits'] as num?)?.toInt() ?? 0,
      reliability: (map['reliability'] as num?)?.toDouble() ?? 1.0,
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'username': username,
      'credits': credits,
      'reliability': reliability,
    };
  }

  UserProfile copyWith({
    String? id,
    String? username,
    DateTime? createdAt,
    int? credits,
    double? reliability,
  }) {
    return UserProfile(
      id: id ?? this.id,
      username: username ?? this.username,
      createdAt: createdAt ?? this.createdAt,
      credits: credits ?? this.credits,
      reliability: reliability ?? this.reliability,
    );
  }
}
