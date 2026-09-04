import 'package:flutter/material.dart';

/// Human Server Design System
/// 
/// Metaphor: Retro telegraph / human-operated computing system.
/// Aesthetic: Parchment, ink, dark walnut, antique brushed brass, and mechanical instruments.
/// Tone: Institutional, understated, tactile, and quiet luxury ("old money").
class AppTheme {
  AppTheme._();

  // Foundation Colors
  static const Color background = Color(0xFF0F1318);       // Deep archival charcoal ink
  static const Color surface = Color(0xFF171D25);          // Dark walnut switchboard panel
  static const Color surfaceElevated = Color(0xFF1F2732);  // Recessed instrument plate
  static const Color surfaceHighlight = Color(0xFF26303D); // Inset compartment
  
  // Paper & Archival Slips
  static const Color parchment = Color(0xFFF2ECE1);        // Telegram paper slip
  static const Color parchmentDark = Color(0xFFE3DAC9);    // Aged document border
  static const Color ink = Color(0xFF1A1715);              // Typewriter ribbon ink
  static const Color inkFaded = Color(0xFF4A443F);         // Muted stamp text
  static const Color inkStamp = Color(0xFF8B2500);         // Wax / red ink verification stamp

  // Hardware & Metal Accents
  static const Color brass = Color(0xFFC5A059);            // Antique brushed brass
  static const Color brassLight = Color(0xFFE2C487);       // Polished brass reflection
  static const Color brassDark = Color(0xFF876A32);        // Tarnished bronze / shadow
  static const Color steel = Color(0xFF4A5568);            // Cold mechanical frame
  static const Color border = Color(0xFF2D3848);           // Machined metal seam
  static const Color borderBrass = Color(0xFF5A482A);      // Engraved brass trim

  // Instrument Signals & State Lamps (Tactile, jewel-lamp hues)
  static const Color signalOnline = Color(0xFF2E8B57);     // Telegraph emerald relay lamp
  static const Color signalOffline = Color(0xFFA93226);    // De-energized crimson filament
  static const Color signalWarning = Color(0xFFD9822B);    // Calibrated amber indicator
  static const Color signalActive = Color(0xFFB8860B);     // Engaged circuit gold

  // Archival Text Hierarchy
  static const Color textPrimary = Color(0xFFEDE8DF);      // Aged ivory readout
  static const Color textSecondary = Color(0xFFA49E93);    // Muted archival annotation
  static const Color textMuted = Color(0xFF6E685E);        // Engraved plate metadata

  // Typography Constants
  static const String serifFamily = 'Georgia';
  static const List<String> serifFallbacks = ['Times New Roman', 'serif'];

  static const String monoFamily = 'Courier New';
  static const List<String> monoFallbacks = ['Consolas', 'monospace'];

  static TextStyle get titleSerif => const TextStyle(
    fontFamily: serifFamily,
    fontFamilyFallback: serifFallbacks,
    color: textPrimary,
    fontWeight: FontWeight.w700,
    letterSpacing: 0.8,
  );

  static TextStyle get telegramMono => const TextStyle(
    fontFamily: monoFamily,
    fontFamilyFallback: monoFallbacks,
    letterSpacing: 0.5,
  );

  static ThemeData get darkTheme {
    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      scaffoldBackgroundColor: background,
      colorScheme: const ColorScheme.dark(
        surface: surface,
        primary: brass,
        secondary: signalOnline,
        error: signalOffline,
        onSurface: textPrimary,
        onPrimary: ink,
      ),
      appBarTheme: const AppBarTheme(
        backgroundColor: background,
        elevation: 0,
        centerTitle: false,
        scrolledUnderElevation: 0,
        titleTextStyle: TextStyle(
          fontFamily: serifFamily,
          fontFamilyFallback: serifFallbacks,
          color: textPrimary,
          fontSize: 20,
          fontWeight: FontWeight.w700,
          letterSpacing: 1.2,
        ),
        iconTheme: IconThemeData(color: brassLight),
      ),
      cardTheme: CardTheme(
        color: surface,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(4), // Square/modest bevel, not bubbly cards
          side: const BorderSide(color: border, width: 1),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: surfaceElevated,
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
        hintStyle: const TextStyle(
          color: textMuted,
          fontSize: 14,
          fontFamily: monoFamily,
          fontFamilyFallback: monoFallbacks,
        ),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(3),
          borderSide: const BorderSide(color: border, width: 1.2),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(3),
          borderSide: const BorderSide(color: border, width: 1.2),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(3),
          borderSide: const BorderSide(color: brass, width: 1.5),
        ),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: surfaceElevated,
          foregroundColor: brassLight,
          elevation: 0,
          padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 15),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(3),
            side: const BorderSide(color: brassDark, width: 1.2),
          ),
          textStyle: const TextStyle(
            fontFamily: serifFamily,
            fontFamilyFallback: serifFallbacks,
            fontSize: 14,
            fontWeight: FontWeight.w700,
            letterSpacing: 1.2,
          ),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: textPrimary,
          side: const BorderSide(color: borderBrass, width: 1),
          padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 13),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(3),
          ),
          textStyle: const TextStyle(
            fontFamily: monoFamily,
            fontFamilyFallback: monoFallbacks,
            fontSize: 12,
            fontWeight: FontWeight.w600,
            letterSpacing: 0.8,
          ),
        ),
      ),
      bottomNavigationBarTheme: const BottomNavigationBarThemeData(
        backgroundColor: surface,
        selectedItemColor: brassLight,
        unselectedItemColor: textMuted,
        type: BottomNavigationBarType.fixed,
        elevation: 0,
        selectedLabelStyle: TextStyle(
          fontFamily: serifFamily,
          fontFamilyFallback: serifFallbacks,
          fontSize: 11,
          fontWeight: FontWeight.w700,
          letterSpacing: 0.8,
        ),
        unselectedLabelStyle: TextStyle(
          fontFamily: serifFamily,
          fontFamilyFallback: serifFallbacks,
          fontSize: 11,
          fontWeight: FontWeight.w500,
        ),
      ),
      dividerTheme: const DividerThemeData(
        color: border,
        thickness: 1,
        space: 24,
      ),
    );
  }
}
