class RecalledFragmentItem {
  final String fragmentId;
  final int sequenceNumber;
  final String recalledText;
  final DateTime createdAt;

  const RecalledFragmentItem({
    required this.fragmentId,
    required this.sequenceNumber,
    required this.recalledText,
    required this.createdAt,
  });

  factory RecalledFragmentItem.fromMap(Map<String, dynamic> map) {
    return RecalledFragmentItem(
      fragmentId: map['fragment_id'] as String? ?? '',
      sequenceNumber: (map['sequence_number'] as num?)?.toInt() ?? 0,
      recalledText: map['recalled_text'] as String? ?? '',
      createdAt: map['created_at'] != null
          ? DateTime.tryParse(map['created_at'].toString()) ?? DateTime.now()
          : DateTime.now(),
    );
  }
  String get fragmentText => recalledText;
}

class RetrievalStatusResult {
  final String retrievalId;
  final String memoryId;
  final String status;
  final int totalFragments;
  final int recalledCount;
  final List<RecalledFragmentItem> fragments;

  const RetrievalStatusResult({
    required this.retrievalId,
    required this.memoryId,
    required this.status,
    required this.totalFragments,
    required this.recalledCount,
    required this.fragments,
  });

  factory RetrievalStatusResult.fromMap(Map<String, dynamic> map) {
    final list = (map['responses'] as List<dynamic>?) ?? [];
    return RetrievalStatusResult(
      retrievalId: map['retrieval_id'] as String? ?? '',
      memoryId: map['memory_id'] as String? ?? '',
      status: map['status'] as String? ?? 'open',
      totalFragments: (map['total_fragments'] as num?)?.toInt() ?? 0,
      recalledCount: (map['recalled_fragments_count'] as num?)?.toInt() ?? list.length,
      fragments: list
          .map((item) => RecalledFragmentItem.fromMap(Map<String, dynamic>.from(item as Map)))
          .toList(),
    );
  }

  int get hostingNodesCount => totalFragments;
  bool get isComplete => status == 'completed' || (totalFragments > 0 && recalledCount >= totalFragments);
}
