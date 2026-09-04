import 'dart:math' as math;
import 'package:flutter/material.dart';
import '../theme/app_theme.dart';

/// Interactive visual representation of the human node communications network.
/// Replaces explanatory copy with an instrument-style network topology map.
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
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(2),
        border: Border.all(color: AppTheme.borderBrass, width: 1.2),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.5),
            blurRadius: 6,
            offset: const Offset(0, 3),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header Bar with Instrument Title & Gauge
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'TELEGRAPH RELAY TOPOLOGY',
                    style: TextStyle(
                      fontFamily: AppTheme.serifFamily,
                      fontFamilyFallback: AppTheme.serifFallbacks,
                      color: AppTheme.brassLight,
                      fontSize: 13,
                      fontWeight: FontWeight.w700,
                      letterSpacing: 1.2,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    '${widget.activeNodesCount} ACTIVE HUMAN STATIONS IN CIRCUIT',
                    style: const TextStyle(
                      fontFamily: AppTheme.monoFamily,
                      fontFamilyFallback: AppTheme.monoFallbacks,
                      color: AppTheme.textMuted,
                      fontSize: 10,
                      letterSpacing: 0.8,
                    ),
                  ),
                ],
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: AppTheme.surfaceElevated,
                  border: Border.all(color: AppTheme.borderBrass),
                  borderRadius: BorderRadius.circular(2),
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
                    const SizedBox(width: 6),
                    Text(
                      '${widget.networkHealth.toStringAsFixed(1)}% COHESION',
                      style: const TextStyle(
                        fontFamily: AppTheme.monoFamily,
                        fontFamilyFallback: AppTheme.monoFallbacks,
                        color: AppTheme.textPrimary,
                        fontSize: 10,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),

          const SizedBox(height: 16),

          // Custom Painted Network Grid
          AnimatedBuilder(
            animation: _pulseController,
            builder: (context, child) {
              return SizedBox(
                height: 160,
                width: double.infinity,
                child: CustomPaint(
                  painter: _NetworkTopologyPainter(
                    pulseValue: _pulseController.value,
                    nodesCount: widget.activeNodesCount,
                  ),
                ),
              );
            },
          ),

          const SizedBox(height: 14),

          // Footer Telemetry Strip
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: BoxDecoration(
              color: AppTheme.surfaceElevated,
              borderRadius: BorderRadius.circular(2),
              border: Border.all(color: AppTheme.border),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                _buildStatusTick('DISPATCH: READY', AppTheme.signalOnline),
                _buildStatusTick('PARITY: 3X REDUNDANT', AppTheme.brass),
                _buildStatusTick('STORAGE: BIOLOGICAL', AppTheme.textSecondary),
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
        Text(
          text,
          style: TextStyle(
            fontFamily: AppTheme.monoFamily,
            fontFamilyFallback: AppTheme.monoFallbacks,
            color: AppTheme.textSecondary,
            fontSize: 9.5,
            letterSpacing: 0.6,
          ),
        ),
      ],
    );
  }
}

class _NetworkTopologyPainter extends CustomPainter {
  final double pulseValue;
  final int nodesCount;

  _NetworkTopologyPainter({
    required this.pulseValue,
    required this.nodesCount,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height * 0.45);

    // Line styles for physical telegraph lines
    final linePaint = Paint()
      ..color = AppTheme.borderBrass.withValues(alpha: 0.7)
      ..strokeWidth = 1.2
      ..style = PaintingStyle.stroke;

    final pulsePaint = Paint()
      ..color = AppTheme.brassLight
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;

    // Calculate satellite stations in a radial orbit
    final totalNodes = math.max(nodesCount, 5);
    final radiusX = size.width * 0.38;
    final radiusY = size.height * 0.38;

    final List<Offset> nodePositions = [];
    final List<String> nodeLabels = [
      'STATION 014',
      'STATION 021',
      'STATION 033',
      'STATION 047',
      'STATION 058',
      'STATION 062',
      'STATION 077',
      'STATION 089',
    ];

    for (int i = 0; i < totalNodes; i++) {
      final angle = (i * (2 * math.pi / totalNodes)) - (math.pi / 2);
      final x = center.dx + (radiusX * math.cos(angle));
      final y = center.dy + (radiusY * math.sin(angle));
      nodePositions.add(Offset(x, y));
    }

    // Draw telegraph connection lines from center to each node
    for (int i = 0; i < nodePositions.length; i++) {
      final target = nodePositions[i];
      canvas.drawLine(center, target, linePaint);

      // Pulse traveling along the wire
      final pulsePos = Offset.lerp(center, target, (pulseValue + (i / totalNodes)) % 1.0)!;
      canvas.drawCircle(pulsePos, 2.5, pulsePaint);
    }

    // Draw Peripheral Human Node Stations
    for (int i = 0; i < nodePositions.length; i++) {
      final pos = nodePositions[i];
      final isOnline = i % 6 != 4; // Mock one station being degraded occasionally

      // Outer bezel plate
      final plateRect = Rect.fromCenter(center: pos, width: 34, height: 20);
      final platePaint = Paint()
        ..color = AppTheme.surfaceElevated
        ..style = PaintingStyle.fill;
      final borderPaint = Paint()
        ..color = isOnline ? AppTheme.borderBrass : AppTheme.signalWarning
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.0;

      canvas.drawRect(plateRect, platePaint);
      canvas.drawRect(plateRect, borderPaint);

      // Center signal jewel
      final jewelPaint = Paint()
        ..color = isOnline ? AppTheme.signalOnline : AppTheme.signalWarning
        ..style = PaintingStyle.fill;
      canvas.drawCircle(Offset(pos.dx - 8, pos.dy), 3, jewelPaint);

      // Node number text
      final textSpan = TextSpan(
        text: (i + 1).toString().padLeft(2, '0'),
        style: TextStyle(
          fontFamily: AppTheme.monoFamily,
          fontFamilyFallback: AppTheme.monoFallbacks,
          color: AppTheme.textPrimary,
          fontSize: 9,
          fontWeight: FontWeight.bold,
        ),
      );
      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      )..layout();
      textPainter.paint(canvas, Offset(pos.dx - 1, pos.dy - (textPainter.height / 2)));
    }

    // Draw Central Hub Station (Your Terminal / Dispatch Key)
    final hubRect = Rect.fromCenter(center: center, width: 84, height: 32);
    final hubPaint = Paint()
      ..color = AppTheme.surfaceElevated
      ..style = PaintingStyle.fill;
    final hubBorder = Paint()
      ..color = AppTheme.brass
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5;

    canvas.drawRect(hubRect, hubPaint);
    canvas.drawRect(hubRect, hubBorder);

    // Hub label
    final hubText = TextPainter(
      text: const TextSpan(
        text: 'CENTRAL\nDISPATCH',
        style: TextStyle(
          fontFamily: AppTheme.serifFamily,
          fontFamilyFallback: AppTheme.serifFallbacks,
          color: AppTheme.brassLight,
          fontSize: 9,
          fontWeight: FontWeight.w800,
          height: 1.1,
          letterSpacing: 0.8,
        ),
      ),
      textAlign: TextAlign.center,
      textDirection: TextDirection.ltr,
    )..layout();
    hubText.paint(canvas, Offset(center.dx - (hubText.width / 2), center.dy - (hubText.height / 2)));
  }

  @override
  bool shouldRepaint(covariant _NetworkTopologyPainter oldDelegate) {
    return oldDelegate.pulseValue != pulseValue || oldDelegate.nodesCount != nodesCount;
  }
}
