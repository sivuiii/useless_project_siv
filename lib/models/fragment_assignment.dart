class FragmentAssignmentModel {
  final String id;
  final String fragmentId;
  final String? nodeId;
  final int replicaNumber;
  final DateTime assignedAt;
  final String status; // 'assigned', 'fulfilled', 'unavailable', 'pending'
  final DateTime? lastVerifiedAt;

  const FragmentAssignmentModel({
    required this.id,
    required this.fragmentId,
    this.nodeId,
    required this.replicaNumber,
    required this.assignedAt,
    required this.status,
    this.lastVerifiedAt,
  });

  bool get isPending => status == 'pending';
  bool get isAssigned => status == 'assigned';

  factory FragmentAssignmentModel.fromMap(Map<String, dynamic> map) {
    return FragmentAssignmentModel(
      id: map['id'] as String? ?? '',
      fragmentId: map['fragment_id'] as String? ?? '',
      nodeId: map['node_id'] as String?,
      replicaNumber: (map['replica_number'] as num?)?.toInt() ?? 1,
      assignedAt: map['assigned_at'] != null
          ? DateTime.tryParse(map['assigned_at'].toString()) ?? DateTime.now()
          : DateTime.now(),
      status: map['status'] as String? ?? 'assigned',
      lastVerifiedAt: map['last_verified_at'] != null
          ? DateTime.tryParse(map['last_verified_at'].toString())
          : null,
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'fragment_id': fragmentId,
      if (nodeId != null) 'node_id': nodeId,
      'replica_number': replicaNumber,
      'assigned_at': assignedAt.toIso8601String(),
      'status': status,
      if (lastVerifiedAt != null)
        'last_verified_at': lastVerifiedAt!.toIso8601String(),
    };
  }
}

/// Represents a fragment hosted on this node (returned by get_my_node_assigned_fragments RPC)
class AssignedNodeFragment {
  final String fragmentId;
  final int sequenceNumber;
  final String fragmentText;
  final int sizeBytes;
  final DateTime expiresAt;
  final String assignmentId;
  final String assignmentStatus;
  final DateTime assignedAt;

  const AssignedNodeFragment({
    required this.fragmentId,
    required this.sequenceNumber,
    required this.fragmentText,
    required this.sizeBytes,
    required this.expiresAt,
    required this.assignmentId,
    required this.assignmentStatus,
    required this.assignedAt,
  });

  factory AssignedNodeFragment.fromMap(Map<String, dynamic> map) {
    return AssignedNodeFragment(
      fragmentId: map['fragment_id'] as String? ?? '',
      sequenceNumber: (map['sequence_number'] as num?)?.toInt() ?? 1,
      fragmentText: map['fragment_text'] as String? ?? '',
      sizeBytes: (map['size_bytes'] as num?)?.toInt() ?? 0,
      expiresAt: map['expires_at'] != null
          ? DateTime.tryParse(map['expires_at'].toString()) ?? DateTime.now()
          : DateTime.now(),
      assignmentId: map['assignment_id'] as String? ?? '',
      assignmentStatus: map['assignment_status'] as String? ?? 'assigned',
      assignedAt: map['assigned_at'] != null
          ? DateTime.tryParse(map['assigned_at'].toString()) ?? DateTime.now()
          : DateTime.now(),
    );
  }
}
