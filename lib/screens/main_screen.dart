import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../services/node_service.dart';
import '../theme/app_theme.dart';
import 'home_screen.dart';
import 'node_screen.dart';
import 'profile_screen.dart';
import 'retrieve_screen.dart';
import 'store_screen.dart';

class MainScreen extends StatefulWidget {
  const MainScreen({super.key});

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  int _currentIndex = 0;

  void _onTabSelected(int index) {
    setState(() {
      _currentIndex = index;
    });

    if (index == 3) {
      NodeService.instance.syncNode();
    }
  }

  @override
  Widget build(BuildContext context) {
    final List<Widget> screens = [
      HomeScreen(
        onNavigateToStore: () => _onTabSelected(1),
        onNavigateToRetrieve: () => _onTabSelected(2),
      ),
      const StoreScreen(),
      const RetrieveScreen(),
      const NodeScreen(),
      const ProfileScreen(),
    ];

    return Scaffold(
      body: IndexedStack(
        index: _currentIndex,
        children: screens,
      ),
      bottomNavigationBar: Container(
        decoration: const BoxDecoration(
          color: AppTheme.surface,
          border: Border(top: BorderSide(color: AppTheme.border, width: 0.8)),
        ),
        child: SafeArea(
          top: false,
          child: Theme(
            data: Theme.of(context).copyWith(
              splashColor: Colors.transparent,
              highlightColor: Colors.transparent,
            ),
            child: BottomNavigationBar(
              currentIndex: _currentIndex,
              onTap: _onTabSelected,
              backgroundColor: AppTheme.surface,
              selectedItemColor: AppTheme.brass,
              unselectedItemColor: AppTheme.textMuted,
              type: BottomNavigationBarType.fixed,
              elevation: 0,
              iconSize: 20,
              selectedFontSize: 10,
              unselectedFontSize: 10,
              selectedLabelStyle: GoogleFonts.inter(
                fontWeight: FontWeight.w600,
                height: 1.4,
              ),
              unselectedLabelStyle: GoogleFonts.inter(
                fontWeight: FontWeight.w500,
                height: 1.4,
              ),
              items: const [
                BottomNavigationBarItem(
                  icon: Icon(Icons.hub_outlined),
                  activeIcon: Icon(Icons.hub_rounded),
                  label: 'Desk',
                ),
                BottomNavigationBarItem(
                  icon: Icon(Icons.send_outlined),
                  activeIcon: Icon(Icons.send_rounded),
                  label: 'Dispatch',
                ),
                BottomNavigationBarItem(
                  icon: Icon(Icons.downloading_outlined),
                  activeIcon: Icon(Icons.downloading_rounded),
                  label: 'Recall',
                ),
                BottomNavigationBarItem(
                  icon: Icon(Icons.dns_outlined),
                  activeIcon: Icon(Icons.dns_rounded),
                  label: 'Station',
                ),
                BottomNavigationBarItem(
                  icon: Icon(Icons.badge_outlined),
                  activeIcon: Icon(Icons.badge_rounded),
                  label: 'Register',
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
