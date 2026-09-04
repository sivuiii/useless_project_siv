import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme/app_theme.dart';

/// Responsive visual representation of the human node communications network.
/// Automatically adapts between compact mobile aspect ratios and expansive desktop views.
class TelegraphNetworkMap extends StatefulWidget {
  final int activeNodesCount;
  final double networkHealth;
  final String? activeMemoryTitle;
  final int fragmentsCount;

  const TelegraphNetworkMap({
    super.key,
    this.activeNodesCount = 6,
    this.networkHealth = 96.4,
    this.activeMemoryTitle,
    this.fragmentsCount = 3,
  });

  @override
  State<TelegraphNetworkMap> createState() => _TelegraphNetworkMapState();
}

class _TelegraphNetworkMapState extends State<TelegraphNetworkMap>
    with SingleTickerProviderStateMixin {
  late AnimationController _pulseController;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 4),
    )..repeat();
  }

  @override
  void dispose() {
    _pulseController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(2),
        border: Border.all(color: AppTheme.border, width: 1.0),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header: Title & Network Cohesion
          Row(
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'NETWORK TOPOLOGY',
                      style: GoogleFonts.playfairDisplay(
                        color: AppTheme.textPrimary,
                        fontSize: 14,
                        fontWeight: FontWeight.w700,
                        letterSpacing: 0.5,
                      ),
                    ),
                    const SizedBox(height: 1),
                    Text(
                      '${widget.activeNodesCount} HUMAN NODES IN CIRCUIT',
                      style: GoogleFonts.spaceMono(
                        color: AppTheme.textMuted,
                        fontSize: 9.5,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 3),
                decoration: BoxDecoration(
                  color: AppTheme.surfaceElevated,
                  border: Border.all(color: AppTheme.border),
                  borderRadius: BorderRadius.circular(2),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Container(
                      width: 5,
                      height: 5,
                      decoration: const BoxDecoration(
                        color: AppTheme.signalOnline,
                        shape: BoxShape.circle,
                      ),
                    ),
                    const SizedBox(width: 5),
                    Text(
                      '${widget.networkHealth.toStringAsFixed(1)}% COHESION',
                      style: GoogleFonts.spaceMono(
                        color: AppTheme.textPrimary,
                        fontSize: 9.5,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),

          const SizedBox(height: 12),

          // Responsive Canvas
          LayoutBuilder(
            builder: (context, constraints) {
              final canvasHeight = constraints.maxWidth < 400 ? 140.0 : 170.0;
              return AnimatedBuilder(
                animation: _pulseController,
                builder: (context, child) {
                  return SizedBox(
                    height: canvasHeight,
                    width: constraints.maxWidth,
                    child: CustomPaint(
                      painter: _NetworkTopologyPainter(
                        pulseValue: _pulseController.value,
                        nodesCount: widget.activeNodesCount,
                        isCompact: constraints.maxWidth < 400,
                      ),
                    ),
                  );
                },
              );
            },
          ),

          const SizedBox(height: 10),

          // Subdued Telemetry Strip
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            decoration: BoxDecoration(
              color: AppTheme.surfaceElevated,
              borderRadius: BorderRadius.circular(3),
              border: Border.all(color: AppTheme.border, width: 0.8),
            ),
            child: Row(
              children: [
                Expanded(child: _buildStatusTick('PARITY: 3X REDUNDANT', AppTheme.brass)),
                const SizedBox(width: 8),
                Expanded(child: _buildStatusTick('STORAGE: BIOLOGICAL', AppTheme.signalOnline)),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatusTick(String text, Color color) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 5,
          height: 5,
          decoration: BoxDecoration(color: color, shape: BoxShape.circle),
        ),
        const SizedBox(width: 5),
        Flexible(
          child: Text(
            text,
            style: GoogleFonts.spaceMono(
              color: AppTheme.textSecondary,
              fontSize: 9.5,
              fontWeight: FontWeight.w600,
            ),
            overflow: TextOverflow.ellipsis,
          ),
        ),
      ],
    );
  }
}

class _NetworkTopologyPainter extends CustomPainter {
  final double pulseValue;
  final int nodesCount;
  final bool isCompact;

  _NetworkTopologyPainter({
    required this.pulseValue,
    required this.nodesCount,
    required this.isCompact,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height * 0.48);

    final linePaint = Paint()
      ..color = AppTheme.border.withValues(alpha: 0.9)
      ..strokeWidth = 1.0
      ..style = PaintingStyle.stroke;

    final pulsePaint = Paint()
      ..color = AppTheme.brassLight
      ..strokeWidth = 1.6
      ..style = PaintingStyle.stroke;

    final totalNodes = math.max(nodesCount, 4);
    
    // Safely constrain radius inside canvas bounds so nodes never clip
    final radiusX = math.min(size.width * 0.38, size.width / 2 - (isCompact ? 28 : 34));
    final radiusY = math.min(size.height * 0.38, size.height / 2 - 16);

    final List<Offset> nodePositions = [];

    for (int i = 0; i < totalNodes; i++) {
      final angle = (i * (2 * math.pi / totalNodes)) - (math.pi / 2);
      final x = center.dx + (radiusX * math.cos(angle));
      final y = center.dy + (radiusY * math.sin(angle));
      nodePositions.add(Offset(x, y));
    }

    // Draw telegraph connection lines
    for (int i = 0; i < nodePositions.length; i++) {
      final target = nodePositions[i];
      canvas.drawLine(center, target, linePaint);

      final pulsePos = Offset.lerp(center, target, (pulseValue + (i / totalNodes)) % 1.0)!;
      canvas.drawCircle(pulsePos, 2.0, pulsePaint);
    }

    // Draw Satellite Station Nodes (Tactile pins with jewel light)
    for (int i = 0; i < nodePositions.length; i++) {
      final pos = nodePositions[i];
      final isOnline = i % 6 != 4;

      final pinRadius = isCompact ? 10.0 : 12.0;

      // Outer pin bezel
      final bezelPaint = Paint()
        ..color = AppTheme.surfaceElevated
        ..style = PaintingStyle.fill;
      final borderPaint = Paint()
        ..color = isOnline ? AppTheme.border : AppTheme.signalWarning
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.0;

      canvas.drawCircle(pos, pinRadius, bezelPaint);
      canvas.drawCircle(pos, pinRadius, borderPaint);

      // Center jewel light
      final jewelPaint = Paint()
        ..color = isOnline ? AppTheme.signalOnline : AppTheme.signalWarning
        ..style = PaintingStyle.fill;
      canvas.drawCircle(pos, isCompact ? 2.5 : 3.0, jewelPaint);

      // Station index badge
      final textSpan = TextSpan(
        text: 'N${(i + 1).toString().padLeft(2, '0')}',
        style: GoogleFonts.spaceMono(
          color: AppTheme.textMuted,
          fontSize: isCompact ? 7.5 : 8.5,
          fontWeight: FontWeight.w700,
        ),
      );
      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      )..layout();
      textPainter.paint(
        canvas,
        Offset(pos.dx - (textPainter.width / 2), pos.dy + (pinRadius + 2)),
      );
    }

    // Central Dispatch Plate
    final hubWidth = isCompact ? 68.0 : 80.0;
    final hubHeight = isCompact ? 26.0 : 30.0;
    final hubRect = Rect.fromCenter(center: center, width: hubWidth, height: hubHeight);

    final hubPaint = Paint()
      ..color = AppTheme.surfaceElevated
      ..style = PaintingStyle.fill;
    final hubBorder = Paint()
      ..color = AppTheme.brass.withValues(alpha: 0.6)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.2;

    canvas.drawRect(hubRect, hubPaint);
    canvas.drawRect(hubRect, hubBorder);

    final hubText = TextPainter(
      text: TextSpan(
        text: 'DISPATCH',
        style: GoogleFonts.playfairDisplay(
          color: AppTheme.brassLight,
          fontSize: isCompact ? 8.5 : 9.5,
          fontWeight: FontWeight.w800,
          letterSpacing: 0.6,
        ),
      ),
      textAlign: TextAlign.center,
      textDirection: TextDirection.ltr,
    )..layout();
    hubText.paint(
      canvas,
      Offset(center.dx - (hubText.width / 2), center.dy - (hubText.height / 2)),
    );
  }

  @override
  bool shouldRepaint(covariant _NetworkTopologyPainter oldDelegate) {
    return oldDelegate.pulseValue != pulseValue ||
        oldDelegate.nodesCount != nodesCount ||
        oldDelegate.isCompact != isCompact;
  }
}
