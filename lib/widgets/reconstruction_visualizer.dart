import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme/app_theme.dart';

/// Animated reconstruction visualizer showing human memory stations responding
/// and signals converging into the recovered memory telegram.
class ReconstructionVisualizer extends StatefulWidget {
  final String queryPrompt;
  final String reconstructedContent;
  final double confidenceScore;
  final int respondingNodesCount;
  final String storedDate;
  final String expiresDate;

  const ReconstructionVisualizer({
    super.key,
    required this.queryPrompt,
    required this.reconstructedContent,
    required this.confidenceScore,
    required this.respondingNodesCount,
    required this.storedDate,
    required this.expiresDate,
  });

  @override
  State<ReconstructionVisualizer> createState() => _ReconstructionVisualizerState();
}

class _ReconstructionVisualizerState extends State<ReconstructionVisualizer>
    with SingleTickerProviderStateMixin {
  late AnimationController _animController;
  int _revealedNodesStep = 0;

  @override
  void initState() {
    super.initState();
    _animController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1400),
    );

    _animController.addListener(() {
      final step = (_animController.value * 4).floor();
      if (step != _revealedNodesStep && mounted) {
        setState(() {
          _revealedNodesStep = step;
        });
      }
    });

    _animController.forward();
  }

  @override
  void dispose() {
    _animController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final nodesList = [
      {'name': 'Station 014 (Sivi)', 'status': 'RECEIVED'},
      {'name': 'Station 021 (Alex)', 'status': 'RECEIVED'},
      {'name': 'Station 047 (Marcus)', 'status': 'RECEIVED'},
      {'name': 'Station 058 (Elena)', 'status': 'VERIFIED'},
    ];

    return Container(
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(2),
        border: Border.all(color: AppTheme.border, width: 1.0),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header Bar: Station Call Sequence
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
            decoration: const BoxDecoration(
              color: AppTheme.surfaceElevated,
              border: Border(bottom: BorderSide(color: AppTheme.border, width: 0.8)),
            ),
            child: Row(
              children: [
                Container(
                  width: 6,
                  height: 6,
                  decoration: const BoxDecoration(
                    color: AppTheme.signalOnline,
                    shape: BoxShape.circle,
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    'SIGNAL RECONSTRUCTION',
                    style: GoogleFonts.inter(
                      color: AppTheme.textPrimary,
                      fontSize: 11.5,
                      fontWeight: FontWeight.w700,
                      letterSpacing: 0.4,
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
                const SizedBox(width: 6),
                Text(
                  '${(widget.confidenceScore * 100).toStringAsFixed(1)}% PARITY',
                  style: GoogleFonts.spaceMono(
                    color: AppTheme.signalOnline,
                    fontSize: 9.5,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ],
            ),
          ),

          // Station Calling Progress Slip
          Padding(
            padding: const EdgeInsets.all(14),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'STATIONS RESPONDING',
                  style: GoogleFonts.inter(
                    color: AppTheme.textMuted,
                    fontSize: 10,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 6),
                ...List.generate(nodesList.length, (index) {
                  final node = nodesList[index];
                  final isRevealed = index <= _revealedNodesStep;
                  return Padding(
                    padding: const EdgeInsets.only(bottom: 4),
                    child: Row(
                      children: [
                        Icon(
                          isRevealed ? Icons.check_circle_outline_rounded : Icons.radio_button_unchecked_rounded,
                          size: 12,
                          color: isRevealed ? AppTheme.signalOnline : AppTheme.textMuted,
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            node['name']!,
                            style: GoogleFonts.inter(
                              color: isRevealed ? AppTheme.textPrimary : AppTheme.textMuted,
                              fontSize: 11.5,
                              fontWeight: isRevealed ? FontWeight.w600 : FontWeight.w400,
                            ),
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                        Text(
                          isRevealed ? node['status']! : 'WAITING...',
                          style: GoogleFonts.spaceMono(
                            color: isRevealed ? AppTheme.signalOnline : AppTheme.textMuted,
                            fontSize: 9.5,
                          ),
                        ),
                      ],
                    ),
                  );
                }),
                const SizedBox(height: 12),
                const Divider(thickness: 0.8),
                const SizedBox(height: 8),

                // Reconstructed Document Paper Slip
                Container(
                  decoration: BoxDecoration(
                    color: AppTheme.parchment,
                    border: Border.all(color: AppTheme.border, width: 1.0),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                        decoration: const BoxDecoration(
                          color: AppTheme.parchmentDark,
                          border: Border(bottom: BorderSide(color: AppTheme.border, width: 0.8)),
                        ),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text(
                              'RECONSTRUCTED TELEGRAM',
                              style: GoogleFonts.inter(
                                color: AppTheme.ink,
                                fontSize: 10,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                            Text(
                              'VERIFIED',
                              style: GoogleFonts.spaceMono(
                                color: AppTheme.inkStamp,
                                fontSize: 9.5,
                                fontWeight: FontWeight.w800,
                              ),
                            ),
                          ],
                        ),
                      ),
                      Padding(
                        padding: const EdgeInsets.all(12),
                        child: SelectableText(
                          widget.reconstructedContent,
                          style: GoogleFonts.courierPrime(
                            color: AppTheme.ink,
                            fontSize: 13.5,
                            fontWeight: FontWeight.w700,
                            height: 1.45,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),

                const SizedBox(height: 10),

                // Footer Metadata
                Row(
                  children: [
                    Expanded(
                      child: Text(
                        'Stored: ${widget.storedDate} • Expires: ${widget.expiresDate}',
                        style: GoogleFonts.inter(
                          color: AppTheme.textMuted,
                          fontSize: 10,
                        ),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                    const SizedBox(width: 8),
                    Text(
                      '${widget.respondingNodesCount} CONVERGED',
                      style: GoogleFonts.spaceMono(
                        color: AppTheme.textSecondary,
                        fontSize: 9.5,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
