import 'package:flutter/material.dart';
import '../models/mock_data.dart';
import '../theme/app_theme.dart';
import '../widgets/max_width_container.dart';

class RetrieveScreen extends StatefulWidget {
  const RetrieveScreen({super.key});

  @override
  State<RetrieveScreen> createState() => _RetrieveScreenState();
}

class _RetrieveScreenState extends State<RetrieveScreen> {
  final TextEditingController _promptController = TextEditingController();
  bool _isRetrieving = false;
  bool _hasSearched = false;
  List<RecalledMemoryMock> _results = [];

  @override
  void dispose() {
    _promptController.dispose();
    super.dispose();
  }

  void _handleRetrieve() {
    final query = _promptController.text.trim();
    if (query.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please enter a recall prompt.'),
          backgroundColor: AppTheme.warning,
        ),
      );
      return;
    }

    setState(() {
      _isRetrieving = true;
      _hasSearched = false;
    });

    // Simulate node consensus & retrieval network latency
    Future.delayed(const Duration(milliseconds: 1100), () {
      if (!mounted) return;
      setState(() {
        _isRetrieving = false;
        _hasSearched = true;
        _results = MockDataRepository.mockRecalledMemories;
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Retrieve Memory'),
      ),
      body: SingleChildScrollView(
        child: MaxWidthContainer(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Prompt Description Banner
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: AppTheme.surface,
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: AppTheme.border),
                ),
                child: const Row(
                  children: [
                    Icon(Icons.psychology_rounded, color: AppTheme.secondary, size: 24),
                    SizedBox(width: 14),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Semantic Memory Recall',
                            style: TextStyle(
                              color: AppTheme.textPrimary,
                              fontSize: 14,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                          SizedBox(height: 3),
                          Text(
                            'Describe what you want to recall. Human nodes will locate matching fragments and reconstruct your memory.',
                            style: TextStyle(
                              color: AppTheme.textSecondary,
                              fontSize: 12,
                              height: 1.3,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 20),

              // Query Input Box
              const Text(
                'RECALL PROMPT',
                style: TextStyle(
                  color: AppTheme.textMuted,
                  fontSize: 11,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 0.8,
                ),
              ),
              const SizedBox(height: 8),
              TextField(
                controller: _promptController,
                style: const TextStyle(color: AppTheme.textPrimary, fontSize: 14),
                decoration: InputDecoration(
                  hintText: 'e.g. "What was my cabin Wi-Fi password?" or "Guitar serial number"',
                  prefixIcon: const Icon(Icons.search_rounded, color: AppTheme.textMuted),
                  suffixIcon: _promptController.text.isNotEmpty
                      ? IconButton(
                          icon: const Icon(Icons.clear_rounded, size: 18),
                          onPressed: () {
                            _promptController.clear();
                            setState(() {});
                          },
                        )
                      : null,
                ),
                onChanged: (_) => setState(() {}),
                onSubmitted: (_) => _handleRetrieve(),
              ),

              const SizedBox(height: 16),

              // Retrieve CTA Button
              SizedBox(
                width: double.infinity,
                height: 48,
                child: ElevatedButton.icon(
                  onPressed: _isRetrieving ? null : _handleRetrieve,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppTheme.secondary,
                  ),
                  icon: _isRetrieving
                      ? const SizedBox.shrink()
                      : const Icon(Icons.downloading_rounded, size: 20),
                  label: _isRetrieving
                      ? const Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            SizedBox(
                              width: 18,
                              height: 18,
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                                color: Colors.white,
                              ),
                            ),
                            SizedBox(width: 10),
                            Text('Querying Human Nodes...'),
                          ],
                        )
                      : const Text('Retrieve Memory (Cost: 20 CR)'),
                ),
              ),

              const SizedBox(height: 28),

              // Results Area Header
              if (_hasSearched || _results.isNotEmpty) ...[
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    const Text(
                      'RECALLED RESULTS',
                      style: TextStyle(
                        color: AppTheme.textMuted,
                        fontSize: 11,
                        fontWeight: FontWeight.w700,
                        letterSpacing: 0.8,
                      ),
                    ),
                    if (_hasSearched)
                      Text(
                        '${_results.length} match(es) found',
                        style: const TextStyle(
                          color: AppTheme.secondary,
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                  ],
                ),
                const SizedBox(height: 12),
              ],

              // Results List / Mock Recalled Info
              if (_isRetrieving) ...[
                const SizedBox(height: 20),
                Center(
                  child: Column(
                    children: [
                      const CircularProgressIndicator(color: AppTheme.secondary),
                      const SizedBox(height: 14),
                      Text(
                        'Collecting consensus from human nodes...',
                        style: TextStyle(
                          color: AppTheme.textMuted,
                          fontSize: 13,
                        ),
                      ),
                    ],
                  ),
                ),
              ] else if (_hasSearched && _results.isEmpty) ...[
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(24),
                  decoration: BoxDecoration(
                    color: AppTheme.surface,
                    borderRadius: BorderRadius.circular(14),
                    border: Border.all(color: AppTheme.border),
                  ),
                  child: const Column(
                    children: [
                      Icon(Icons.search_off_rounded, color: AppTheme.textMuted, size: 36),
                      SizedBox(height: 10),
                      Text(
                        'No matching memory found',
                        style: TextStyle(
                          color: AppTheme.textPrimary,
                          fontSize: 14,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      SizedBox(height: 4),
                      Text(
                        'Try refining your recall prompt or keyword.',
                        style: TextStyle(color: AppTheme.textMuted, fontSize: 12),
                      ),
                    ],
                  ),
                ),
              ] else if (_hasSearched) ...[
                ..._results.map((item) => _buildResultCard(item)),
              ] else ...[
                // Initial State Hint
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(24),
                  decoration: BoxDecoration(
                    color: AppTheme.surface.withValues(alpha: 0.5),
                    borderRadius: BorderRadius.circular(14),
                    border: Border.all(color: AppTheme.border.withValues(alpha: 0.6)),
                  ),
                  child: const Column(
                    children: [
                      Icon(Icons.find_in_page_rounded, color: AppTheme.textMuted, size: 32),
                      SizedBox(height: 10),
                      Text(
                        'Enter a recall prompt above to search stored memories',
                        style: TextStyle(
                          color: AppTheme.textSecondary,
                          fontSize: 13,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ],
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildResultCard(RecalledMemoryMock item) {
    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppTheme.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Expanded(
                child: Text(
                  item.promptMatch,
                  style: const TextStyle(
                    color: AppTheme.primaryLight,
                    fontSize: 15,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                  color: AppTheme.secondary.withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(6),
                ),
                child: Text(
                  '${(item.confidenceScore * 100).toStringAsFixed(1)}% Match',
                  style: const TextStyle(
                    color: AppTheme.secondary,
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: AppTheme.background,
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: AppTheme.border),
            ),
            child: SelectableText(
              item.content,
              style: const TextStyle(
                color: AppTheme.textPrimary,
                fontSize: 13,
                height: 1.4,
                fontFamily: 'monospace',
              ),
            ),
          ),
          const SizedBox(height: 12),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'Stored: ${item.storedDate} • Expires: ${item.expiresDate}',
                style: const TextStyle(
                  color: AppTheme.textMuted,
                  fontSize: 11,
                ),
              ),
              Row(
                children: [
                  const Icon(Icons.people_rounded, size: 14, color: AppTheme.textMuted),
                  const SizedBox(width: 4),
                  Text(
                    '${item.respondingNodesCount} nodes verified',
                    style: const TextStyle(
                      color: AppTheme.textSecondary,
                      fontSize: 11,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ],
              ),
            ],
          ),
        ],
      ),
    );
  }
}
