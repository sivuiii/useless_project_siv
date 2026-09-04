import 'dart:math' as math;
import 'package:flutter/material.dart';
import '../theme/app_theme.dart';

/// A precision mechanical instrument dial for measuring human node reliability.
class CalibrationDial extends StatelessWidget {
  final double scorePercent; // 0.0 to 100.0
  final double size;
  final String title;

  const CalibrationDial({
    super.key,
    required this.scorePercent,
    this.size = 140,
    this.title = 'RECALL RELIABILITY',
  });

  String get conditionRating {
    if (scorePercent >= 95.0) return 'OPTIMAL';
    if (scorePercent >= 80.0) return 'ACCEPTABLE';
    if (scorePercent >= 60.0) return 'DEGRADED';
    return 'STANDBY';
  }

  Color get conditionColor {
    if (scorePercent >= 95.0) return AppTheme.signalOnline;
    if (scorePercent >= 80.0) return AppTheme.brassLight;
    if (scorePercent >= 60.0) return AppTheme.signalWarning;
    return AppTheme.signalOffline;
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.surfaceElevated,
        borderRadius: BorderRadius.circular(2),
        border: Border.all(color: AppTheme.borderBrass, width: 1.2),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.4),
            blurRadius: 4,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                title.toUpperCase(),
                style: const TextStyle(
                  fontFamily: AppTheme.serifFamily,
                  fontFamilyFallback: AppTheme.serifFallbacks,
                  color: AppTheme.brassLight,
                  fontSize: 11,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 1.2,
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                decoration: BoxDecoration(
                  color: AppTheme.surface,
                  border: Border.all(color: conditionColor.withValues(alpha: 0.7), width: 1),
                  borderRadius: BorderRadius.circular(2),
                ),
                child: Text(
                  conditionRating,
                  style: TextStyle(
                    fontFamily: AppTheme.monoFamily,
                    fontFamilyFallback: AppTheme.monoFallbacks,
                    color: conditionColor,
                    fontSize: 9.5,
                    fontWeight: FontWeight.w700,
                    letterSpacing: 0.8,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          SizedBox(
            width: size,
            height: size * 0.65,
            child: CustomPaint(
              painter: _DialPainter(
                percent: (scorePercent.clamp(0.0, 100.0)) / 100.0,
                accentColor: conditionColor,
              ),
            ),
          ),
          const SizedBox(height: 6),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.baseline,
            textBaseline: TextBaseline.alphabetic,
            children: [
              Text(
                scorePercent.toStringAsFixed(1),
                style: const TextStyle(
                  fontFamily: AppTheme.serifFamily,
                  fontFamilyFallback: AppTheme.serifFallbacks,
                  color: AppTheme.textPrimary,
                  fontSize: 26,
                  fontWeight: FontWeight.w700,
                  letterSpacing: -0.5,
                ),
              ),
              const SizedBox(width: 3),
              const Text(
                '%',
                style: TextStyle(
                  fontFamily: AppTheme.monoFamily,
                  fontFamilyFallback: AppTheme.monoFallbacks,
                  color: AppTheme.brass,
                  fontSize: 14,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ],
          ),
          const SizedBox(height: 2),
          const Text(
            'BIOLOGICAL RETENTION INDEX',
            style: TextStyle(
              fontFamily: AppTheme.monoFamily,
              fontFamilyFallback: AppTheme.monoFallbacks,
              color: AppTheme.textMuted,
              fontSize: 9,
              letterSpacing: 1.0,
            ),
          ),
        ],
      ),
    );
  }
}

class _DialPainter extends CustomPainter {
  final double percent; // 0.0 to 1.0
  final Color accentColor;

  _DialPainter({required this.percent, required this.accentColor});

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height * 0.9);
    final radius = size.width * 0.42;

    const startAngle = math.pi;
    const sweepAngle = math.pi;

    // Background track
    final trackPaint = Paint()
      ..color = AppTheme.border
      ..style = PaintingStyle.stroke
      ..strokeWidth = 4.0;

    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      startAngle,
      sweepAngle,
      false,
      trackPaint,
    );

    // Ticks around the arc
    final tickPaint = Paint()
      ..color = AppTheme.textMuted
      ..strokeWidth = 1.0;

    const tickCount = 10;
    for (int i = 0; i <= tickCount; i++) {
      final angle = startAngle + (sweepAngle * (i / tickCount));
      final innerR = radius - (i % 5 == 0 ? 8 : 4);
      final outerR = radius + 2;

      final p1 = Offset(center.dx + innerR * math.cos(angle), center.dy + innerR * math.sin(angle));
      final p2 = Offset(center.dx + outerR * math.cos(angle), center.dy + outerR * math.sin(angle));

      canvas.drawLine(p1, p2, tickPaint);
    }

    // Active fill arc
    final activePaint = Paint()
      ..color = accentColor
      ..style = PaintingStyle.stroke
      ..strokeWidth = 4.0
      ..strokeCap = StrokeCap.round;

    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      startAngle,
      sweepAngle * percent,
      false,
      activePaint,
    );

    // Needle
    final needleAngle = startAngle + (sweepAngle * percent);
    final needleLength = radius - 4;
    final needleTip = Offset(
      center.dx + needleLength * math.cos(needleAngle),
      center.dy + needleLength * math.sin(needleAngle),
    );

    final needlePaint = Paint()
      ..color = AppTheme.brassLight
      ..strokeWidth = 2.0
      ..strokeCap = StrokeCap.round;

    canvas.drawLine(center, needleTip, needlePaint);

    // Brass center pivot hub
    final hubPaint = Paint()..color = AppTheme.brassDark;
    canvas.drawCircle(center, 6, hubPaint);

    final hubScrew = Paint()..color = AppTheme.brassLight;
    canvas.drawCircle(center, 3, hubScrew);
  }

  @override
  bool shouldRepaint(covariant _DialPainter oldDelegate) {
    return oldDelegate.percent != percent || oldDelegate.accentColor != accentColor;
  }
}
