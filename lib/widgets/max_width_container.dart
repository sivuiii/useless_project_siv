import 'package:flutter/material.dart';

class MaxWidthContainer extends StatelessWidget {
  final Widget child;
  final double maxWidth;
  final EdgeInsetsGeometry padding;

  const MaxWidthContainer({
    super.key,
    required this.child,
    this.maxWidth = 720.0,
    this.padding = const EdgeInsets.symmetric(horizontal: 16.0, vertical: 16.0),
  });

  @override
  Widget build(BuildContext context) {
    return Center(
      child: ConstrainedBox(
        constraints: BoxConstraints(maxWidth: maxWidth),
        child: Padding(
          padding: padding,
          child: child,
        ),
      ),
    );
  }
}
