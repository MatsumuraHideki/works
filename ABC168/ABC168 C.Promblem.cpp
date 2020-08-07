#include <bits/stdc++.h>
using namespace std;

int main() {
  double a, b, h ,m;
  cin >> a >> b >> h >> m;
  
  double PI=3.14159265358979;
  
  double Ans = sqrt(a*a + b*b - 2*a*b*cos(PI / 6.0 * h - 11.0 / 360.0 * PI * m) );
  
  cout << fixed << setprecision(15);
  
  cout << Ans << endl;
}


