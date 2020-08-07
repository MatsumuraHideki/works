#include <bits/stdc++.h>
using namespace std;

int main() {
  int X, A, B;
  cin >> X;

  A = X / 500;
  B = (X % 500) / 5;
  
  int ans;
  ans = A * 1000 + B * 5;
  
  cout << ans << endl;
}
