import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../models/mock_data.dart';
import '../theme/app_theme.dart';
import '../widgets/max_width_container.dart';
import '../widgets/reconstruction_visualizer.dart';

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
        SnackBar(
          content: Text(
            'PLEASE ENTER A RECALL QUERY PROMPT.',
            style: GoogleFonts.spaceMono(fontSize: 12),
          ),
          backgroundColor: AppTheme.signalWarning,
        ),
      );
      return;
    }

    setState(() {
      _isRetrieving = true;
      _hasSearched = false;
    });

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
        title: Text(
          'RECALL SIGNAL & RECONSTRUCT',
          style: GoogleFonts.playfairDisplay(
            color: AppTheme.textPrimary,
            fontSize: 18,
            fontWeight: FontWeight.w800,
            letterSpacing: 1.2,
          ),
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: MaxWidthContainer(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Prompt Description Banner
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: AppTheme.surface,
                  borderRadius: BorderRadius.circular(3),
                  border: Border.all(color: AppTheme.border, width: 1),
                ),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      padding: const EdgeInsets.all(8),
                      decoration: BoxDecoration(
                        color: AppTheme.surfaceElevated,
                        border: Border.all(color: AppTheme.border),
                        borderRadius: BorderRadius.circular(3),
                      ),
                      child: const Icon(Icons.psychology_rounded, color: AppTheme.brassAccent, size: 20),
                    ),
                    const SizedBox(width: 14),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'SEMANTIC RECALL TELEGRAPH',
                            style: GoogleFonts.inter(
                              color: AppTheme.textPrimary,
                              fontSize: 13,
                              fontWeight: FontWeight.w700,
                              letterSpacing: 0.5,
                            ),
                          ),
                          const SizedBox(height: 3),
                          Text(
                            'Submit recall prompt. Biological station nodes holding matching fragments will transmit responses to reconstruct payload.',
                            style: GoogleFonts.inter(
                              color: AppTheme.textSecondary,
                              fontSize: 12,
                              height: 1.35,
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
              Text(
                'RECALL QUERY PROMPT',
                style: GoogleFonts.inter(
                  color: AppTheme.textMuted,
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                  letterSpacing: 0.5,
                ),
              ),
              const SizedBox(height: 8),
              TextField(
                controller: _promptController,
                style: GoogleFonts.inter(color: AppTheme.textPrimary, fontSize: 13),
                decoration: InputDecoration(
                  hintText: 'e.g. "What was my cabin Wi-Fi password?"',
                  prefixIcon: const Icon(Icons.search_rounded, color: AppTheme.textMuted, size: 18),
                  suffixIcon: _promptController.text.isNotEmpty
                      ? IconButton(
                          icon: const Icon(Icons.clear_rounded, size: 16),
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
                    backgroundColor: AppTheme.surfaceElevated,
                    foregroundColor: AppTheme.brassAccent,
                    side: const BorderSide(color: AppTheme.borderSubtle, width: 1.0),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(4)),
                  ),
                  icon: _isRetrieving
                      ? const SizedBox.shrink()
                      : const Icon(Icons.downloading_rounded, size: 18),
                  label: _isRetrieving
                      ? Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            const SizedBox(
                              width: 14,
                              height: 14,
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                                color: AppTheme.brassAccent,
                              ),
                            ),
                            const SizedBox(width: 10),
                            Flexible(
                              child: Text(
                                'CALLING HUMAN STATIONS...',
                                style: GoogleFonts.inter(
                                  fontSize: 12,
                                  fontWeight: FontWeight.w600,
                                  letterSpacing: 0.5,
                                ),
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                          ],
                        )
                      : Text(
                          'QUERY HUMAN NODES (20 CR)',
                          style: GoogleFonts.inter(
                            fontSize: 12,
                            fontWeight: FontWeight.w700,
                            letterSpacing: 0.5,
                          ),
                        ),
                ),
              ),

              const SizedBox(height: 24),

              // Results Area
              if (_isRetrieving) ...[
                const SizedBox(height: 20),
                Center(
                  child: Column(
                    children: [
                      const CircularProgressIndicator(color: AppTheme.brassLight),
                      const SizedBox(height: 14),
                      Text(
                        'COLLECTING CONSENSUS FROM BIOLOGICAL NODES...',
                        style: GoogleFonts.spaceMono(
                          color: AppTheme.textMuted,
                          fontSize: 11,
                          letterSpacing: 0.8,
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
                    borderRadius: BorderRadius.circular(2),
                    border: Border.all(color: AppTheme.border),
                  ),
                  child: Column(
                    children: [
                      const Icon(Icons.search_off_rounded, color: AppTheme.textMuted, size: 36),
                      const SizedBox(height: 10),
                      Text(
                        'NO MATCHING MEMORY FRAGMENTS',
                        style: GoogleFonts.playfairDisplay(
                          color: AppTheme.textPrimary,
                          fontSize: 15,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'Refine query prompt to match biological retention indexing.',
                        style: GoogleFonts.inter(color: AppTheme.textMuted, fontSize: 12),
                      ),
                    ],
                  ),
                ),
              ] else if (_hasSearched) ...[
                ..._results.map((item) => Padding(
                      padding: const EdgeInsets.only(bottom: 16),
                      child: ReconstructionVisualizer(
                        queryPrompt: item.promptMatch,
                        reconstructedContent: item.content,
                        confidenceScore: item.confidenceScore,
                        respondingNodesCount: item.respondingNodesCount,
                        storedDate: item.storedDate,
                        expiresDate: item.expiresDate,
                      ),
                    )),
              ] else ...[
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(24),
                  decoration: BoxDecoration(
                    color: AppTheme.surface,
                    borderRadius: BorderRadius.circular(2),
                    border: Border.all(color: AppTheme.border),
                  ),
                  child: Column(
                    children: [
                      const Icon(Icons.saved_search_rounded, color: AppTheme.textMuted, size: 32),
                      const SizedBox(height: 10),
                      Text(
                        'ENTER RECALL PROMPT TO INITIATE TELEGRAPH QUERY',
                        style: GoogleFonts.spaceMono(
                          color: AppTheme.textSecondary,
                          fontSize: 11,
                          letterSpacing: 0.5,
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
}
