#include <bits/stdc++.h>
using namespace std;
 
int main () {
int64_t X;
  cin >> X;
  
  int64_t sum  = 100;
  int year = 0;
  while(sum < X){
  sum = sum*1.01;
    year++;
  }

  cout << year << endl;
}
