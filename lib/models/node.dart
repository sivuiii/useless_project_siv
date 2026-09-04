class NodeModel {
  final String id;
  final String userId;
  final String status;
  final double reliability;
  final double responseRate;
  final int avgResponseMs;
  final DateTime lastSeen;
  final DateTime createdAt;

  const NodeModel({
    required this.id,
    required this.userId,
    required this.status,
    required this.reliability,
    required this.responseRate,
    required this.avgResponseMs,
    required this.lastSeen,
    required this.createdAt,
  });

  bool get isOnline => status.toLowerCase() == 'online';

  factory NodeModel.fromMap(Map<String, dynamic> map) {
    return NodeModel(
      id: map['id'] as String? ?? '',
      userId: map['user_id'] as String? ?? '',
      status: map['status'] as String? ?? 'online',
      reliability: (map['reliability'] as num?)?.toDouble() ?? 1.0,
      responseRate: (map['response_rate'] as num?)?.toDouble() ?? 1.0,
      avgResponseMs: (map['avg_response_ms'] as num?)?.toInt() ?? 0,
      lastSeen: map['last_seen'] != null
          ? DateTime.tryParse(map['last_seen'].toString()) ?? DateTime.now()
          : DateTime.now(),
      createdAt: map['created_at'] != null
          ? DateTime.tryParse(map['created_at'].toString()) ?? DateTime.now()
          : DateTime.now(),
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'user_id': userId,
      'status': status,
      'reliability': reliability,
      'response_rate': responseRate,
      'avg_response_ms': avgResponseMs,
      'last_seen': lastSeen.toIso8601String(),
    };
  }

  NodeModel copyWith({
    String? id,
    String? userId,
    String? status,
    double? reliability,
    double? responseRate,
    int? avgResponseMs,
    DateTime? lastSeen,
    DateTime? createdAt,
  }) {
    return NodeModel(
      id: id ?? this.id,
      userId: userId ?? this.userId,
      status: status ?? this.status,
      reliability: reliability ?? this.reliability,
      responseRate: responseRate ?? this.responseRate,
      avgResponseMs: avgResponseMs ?? this.avgResponseMs,
      lastSeen: lastSeen ?? this.lastSeen,
      createdAt: createdAt ?? this.createdAt,
    );
  }
}
