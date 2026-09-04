import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

/// Human Server Design System
/// 
/// Metaphor: Retro telegraph communications office / human-operated computing network.
/// Aesthetic: Deep ink, subtle dark walnut, restrained antique brass accents, warm ivory,
/// parchment slips, and jewel signal lamps.
/// Tone: Institutional, understated, tactile, and quiet luxury.
class AppTheme {
  AppTheme._();

  // Foundation Colors
  static const Color background = Color(0xFF0F1216);       // Deep archival ink / charcoal
  static const Color surface = Color(0xFF161B22);          // Dark walnut switchboard desk
  static const Color surfaceElevated = Color(0xFF1E252E);  // Recessed instrument plate
  static const Color surfaceHighlight = Color(0xFF262F3B); // Subtle highlight
  
  // Backward compatibility getters
  static const Color surfaceLight = surfaceElevated;
  static const Color primary = brass;
  static const Color primaryLight = brassLight;
  static const Color secondary = signalOnline;
  static const Color warning = signalWarning;
  static const Color error = signalOffline;

  // Paper & Archival Slips
  static const Color parchment = Color(0xFFF2ECE1);        // Telegram paper slip
  static const Color parchmentDark = Color(0xFFE3DAC9);    // Aged document border
  static const Color ink = Color(0xFF141210);              // Typewriter ribbon ink
  static const Color inkFaded = Color(0xFF4E4740);         // Muted stamp text
  static const Color inkStamp = Color(0xFF8B2500);         // Red wax / verification stamp

  // Hardware & Metal Accents (Gold is an ACCENT, not the default border!)
  static const Color brass = Color(0xFFC5A059);            // Antique brushed brass accent
  static const Color brassAccent = brass;                  // Alias for brass accent
  static const Color brassLight = Color(0xFFE2C487);       // Polished brass highlight
  static const Color brassDark = Color(0xFF876A32);        // Tarnished bronze / shadow
  static const Color steel = Color(0xFF4A5568);            // Cold mechanical frame
  static const Color border = Color(0xFF242B35);           // Subtle dark hairline border (DEFAULT)
  static const Color borderSubtle = border;                // Alias for subtle border
  static const Color borderBrass = Color(0xFF5A482A);      // Selective brass trim (ACCENT ONLY)

  // Instrument Signals & State Lamps (Subtle, jewel-lamp hues)
  static const Color signalOnline = Color(0xFF2E8B57);     // Telegraph emerald relay lamp
  static const Color signalOffline = Color(0xFFA93226);    // De-energized crimson filament
  static const Color signalWarning = Color(0xFFD9822B);    // Calibrated amber indicator
  static const Color signalActive = Color(0xFFB8860B);     // Engaged circuit gold

  // Archival Text Hierarchy
  static const Color textPrimary = Color(0xFFEDE8DF);      // Warm ivory / parchment white
  static const Color textSecondary = Color(0xFF9EA5B0);    // Muted archival annotation
  static const Color textMuted = Color(0xFF6B7280);        // Subdued plate metadata

  // Typography Fallbacks
  static const String serifFamily = 'Georgia';
  static const List<String> serifFallbacks = ['Georgia', 'Times New Roman', 'serif'];

  static const String monoFamily = 'Courier New';
  static const List<String> monoFallbacks = ['Courier New', 'Consolas', 'monospace'];

  // Headings: Sophisticated Editorial Serif
  static TextStyle get titleSerif => GoogleFonts.playfairDisplay(
    color: textPrimary,
    fontWeight: FontWeight.w700,
    letterSpacing: 0.5,
  );

  // Functional UI: Clean, Highly Readable Sans-Serif
  static TextStyle get bodySans => GoogleFonts.inter(
    color: textPrimary,
    fontSize: 14,
    height: 1.45,
  );

  static TextStyle get uiLabel => GoogleFonts.inter(
    color: textSecondary,
    fontSize: 12,
    fontWeight: FontWeight.w600,
  );

  // System Data: Restrained Monospace (Station IDs, timestamps, telegraph payload)
  static TextStyle get telegramMono => GoogleFonts.spaceMono(
    color: textPrimary,
    letterSpacing: 0.3,
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
      textTheme: TextTheme(
        headlineMedium: titleSerif.copyWith(fontSize: 22),
        titleLarge: titleSerif.copyWith(fontSize: 18),
        titleMedium: GoogleFonts.inter(
          color: textPrimary,
          fontSize: 15,
          fontWeight: FontWeight.w600,
        ),
        bodyLarge: GoogleFonts.inter(
          color: textPrimary,
          fontSize: 14,
          height: 1.45,
        ),
        bodyMedium: GoogleFonts.inter(
          color: textSecondary,
          fontSize: 13,
          height: 1.4,
        ),
        labelSmall: GoogleFonts.spaceMono(
          color: textMuted,
          fontSize: 10,
          fontWeight: FontWeight.w600,
          letterSpacing: 0.5,
        ),
      ),
      appBarTheme: AppBarTheme(
        backgroundColor: background,
        elevation: 0,
        centerTitle: false,
        scrolledUnderElevation: 0,
        titleTextStyle: GoogleFonts.playfairDisplay(
          color: textPrimary,
          fontSize: 18,
          fontWeight: FontWeight.w700,
          letterSpacing: 0.8,
        ),
        iconTheme: const IconThemeData(color: textPrimary, size: 20),
      ),
      cardTheme: CardTheme(
        color: surface,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(3),
          side: const BorderSide(color: border, width: 1.0),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: surfaceElevated,
        contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
        hintStyle: GoogleFonts.inter(
          color: textMuted,
          fontSize: 13,
        ),
        labelStyle: GoogleFonts.inter(
          color: textSecondary,
          fontSize: 12,
        ),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(2),
          borderSide: const BorderSide(color: border, width: 1.0),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(2),
          borderSide: const BorderSide(color: border, width: 1.0),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(2),
          borderSide: const BorderSide(color: brass, width: 1.2),
        ),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: surfaceElevated,
          foregroundColor: textPrimary,
          elevation: 0,
          padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 12),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(2),
            side: const BorderSide(color: border, width: 1.0),
          ),
          textStyle: GoogleFonts.inter(
            fontSize: 13,
            fontWeight: FontWeight.w600,
            letterSpacing: 0.3,
          ),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: textPrimary,
          side: const BorderSide(color: border, width: 1.0),
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(2),
          ),
          textStyle: GoogleFonts.inter(
            fontSize: 12,
            fontWeight: FontWeight.w600,
            letterSpacing: 0.3,
          ),
        ),
      ),
      bottomNavigationBarTheme: BottomNavigationBarThemeData(
        backgroundColor: surface,
        selectedItemColor: brassLight,
        unselectedItemColor: textMuted,
        type: BottomNavigationBarType.fixed,
        elevation: 0,
        selectedLabelStyle: GoogleFonts.inter(
          fontSize: 10,
          fontWeight: FontWeight.w600,
        ),
        unselectedLabelStyle: GoogleFonts.inter(
          fontSize: 10,
          fontWeight: FontWeight.w400,
        ),
      ),
      dividerTheme: const DividerThemeData(
        color: border,
        thickness: 1,
        space: 20,
      ),
    );
  }
}
