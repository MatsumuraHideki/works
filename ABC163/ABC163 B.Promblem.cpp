#include <bits/stdc++.h>
using namespace std;

int main() {
  int N, M;
  cin >> N >> M;
  
  vector<int> vec(10000);
  for (int i = 0; i < M; i++) {
    cin >> vec.at(i);
}

  int sum = 0;
  for (int j = 0; j < M; j++){
    sum = sum + vec.at(j);
  }
  
  if (sum > N){
    cout << -1 <<endl;
  }
  else{
    cout << N - sum << endl;
  }
}
