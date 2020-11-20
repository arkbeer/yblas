#include<iostream>
#include<iomanip>
#include"random.hpp"
#include"yblas.hpp"
#include<omp.h>
#include<mpi.h>
#include <sstream>
void RefSyEv(yblas::matrix<double>& A, yblas::matrix<double>& X){
    const int n=X.ncol();
    auto I=yblas::make_diag<double>(n,n);
    auto R=I-X.t()*X;
    auto S=X.t()*A*X;
    yblas::matrix<double> D(X.ncol(),X.ncol());
    D=D.diag();
    for(int i=0;i<X.ncol();++i){
        D.at(i,i)=S.at(i,i)/(1-R.at(i,i));
    }
    double delta=((S-D).norm('F')*2.0/std::sqrt(n)+A.norm('F')*2.0/std::sqrt(n)*R.norm('F')*2.0/std::sqrt(n))*2.0;
    yblas::matrix<double> E(X.ncol(),X.nrow());
    for(int j=0;j<X.nrow();++j){
        for(int i=0;i<X.ncol();++i){
            if(std::abs(D.at(i,i)-D.at(j,j))<=delta){
                E.at(i,j)=R.at(i,j)/2.0;
            }else{
                E.at(i,j)=(S.at(i,j)+D.at(j,j)*R.at(i,j))/(D.at(j,j)-D.at(i,i));
            }
        }
    }
    X=X+X*E;
}
int main(int argc, char **argv){
    yblas y;
    ark::random rand;
    int n=10000;
    
    yblas::matrix<double> da(n,n);
    yblas::matrix<float> fa(n,n);
    if(y.rank()==0){
        for(auto& v: da.data()){
            v=rand.range_real(0,1);
        }
    }
    da=(da+da.t())/2;
    fa=da;
    double tic,toc;
    if(y.rank()==0){
        std::string tmp="";
        tmp=tmp+"n: "+std::to_string(n);
        std::cout<<tmp<<std::endl;
    }

    tic=MPI_Wtime();
    auto fpair=y.syev(fa);
    toc=MPI_Wtime();
    if(y.rank()==0){
        std::string tmp="";
        tmp=tmp+"pssyev_: "+std::to_string(toc-tic);
        std::cout<<tmp<<std::endl;
    }

    yblas::matrix<double> dx(n,n);
    dx=fpair.second;
    tic=MPI_Wtime();
    for(int i=0;i<2;++i){
        RefSyEv(da, dx);
    }
    toc=MPI_Wtime();
    if(y.rank()==0){
        std::string tmp="";
        tmp=tmp+"RefSyEv: "+std::to_string(toc-tic);
        std::cout<<tmp<<std::endl;
    }

    tic=MPI_Wtime();
    auto dpair=y.syev(da);
    toc=MPI_Wtime();
    if(y.rank()==0){
        std::string tmp="";
        tmp=tmp+"pdsyev_: "+std::to_string(toc-tic);
        std::cout<<tmp<<std::endl;
    }
  
    auto result1=(y.make_diag<double>(n,n)-dx.t()*dx).norm('F');
    if(y.rank()==0)std::cout<<result1<<std::endl;
    auto tmp=dx.t()*da*dx;
    for(int i=0;i<n;++i){
        tmp.at(i,i)=0;
    }
    auto result2=tmp.norm('F');
    if(y.rank()==0)std::cout<<result2<<std::endl;
    return 0;
}