#include <bits/stdc++.h>
using namespace std;

int main(){
int N, M;
  cin >> N >> M;
  
  vector<int> H(N);
  
  for(int i=0;i<N;i++){
    cin >> H.at(i);
  }
  
  vector<vector<int>> G(M, vector<int>(2));
  for(int j=0;j<M;j++){
    cin >> G.at(j).at(0) >> G.at(j).at(1);
  }
  
  set<int> MAX;
  set<int> MIN;
  set<int> ALL;
  
  for(int k=0;k<M;k++){
    if(H.at(G.at(k).at(0)-1) > H.at(G.at(k).at(1)-1)){
       MAX.insert(G.at(k).at(0));
       MIN.insert(G.at(k).at(1));
    }
    else if(H.at(G.at(k).at(0)-1) < H.at(G.at(k).at(1)-1)){
       MAX.insert(G.at(k).at(1));
       MIN.insert(G.at(k).at(0));
    }
    else if(H.at(G.at(k).at(0)-1) == H.at(G.at(k).at(1)-1)){
       MIN.insert(G.at(k).at(0));
       MIN.insert(G.at(k).at(1));
    }
      ALL.insert(G.at(k).at(0));
      ALL.insert(G.at(k).at(1));
}
  
  int EX = N - ALL.size();
  
  for(int l=1;l<N+1;l++){
  if(MAX.count(l) && MIN.count(l)){
    MAX.erase(l);
  }
  }
  
  cout << MAX.size() + EX << endl;
}
    
    
