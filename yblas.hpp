#include<vector>
#include<array>
#include<cmath>
#include <mkl_blacs.h>
#include <mkl_pblas.h>
#include <mkl_scalapack.h>
#include <mkl.h>

extern "C" void descinit_ (MKL_INT *desc, const MKL_INT *m, const MKL_INT *n, const MKL_INT *mb, const MKL_INT *nb, const MKL_INT *irsrc, const MKL_INT *icsrc, const MKL_INT *ictxt, const MKL_INT *lld, MKL_INT *info);
extern "C" MKL_INT numroc_ (const MKL_INT *n, const MKL_INT *nb, const MKL_INT *iproc, const MKL_INT *srcproc, const MKL_INT *nprocs);

class yblas{
    public:
    static yblas* ptr;
    int nblock;
    template <typename T>
    class matrix{
        std::vector<T> _data;
        const int _ncol, _nrow;
        public:
        matrix(const int ncol, const int nrow):_data(std::vector<T>(ncol*nrow)),_ncol(ncol),_nrow(nrow){}
        int ncol()const {return _ncol;}
        int nrow()const {return _nrow;}
        auto& data(){return _data;}
        const auto& data()const {return _data;}
        T& at(const int n){return _data.at(n);}
        T& at(const int m, const int n){return _data.at(m+n*ncol());}
        auto t(){
            if(yblas::ptr!=nullptr){
                auto c=ptr->tran(*this);
                return c;
            }
            return *this;
        }
        auto diag(){
            for(int j=0;j<nrow();++j){
                for(int i=0;i<ncol();++i){
                    if(i==j){at(i,j)=1;}else{at(i,j)=0;}
                }
            }
            return *this;
        }
        T norm(const char norm){
            if(yblas::ptr!=nullptr){
                auto c=ptr->lange(norm, *this);
                return c;
            }
            return 0;
        }
        matrix<T>& operator=(const matrix<T>& b){
            if(nrow()==b.nrow() && ncol()==b.ncol())_data=b._data;
            return *this;
        }
        template<typename T2>
        matrix<T>& operator=(const matrix<T2>& b){
            if(nrow()==b.nrow() && ncol()==b.ncol()){
                for(int i=0;i<nrow()*ncol();++i){
                    _data.at(i)=static_cast<T>(b.data().at(i));
                }
            }
            return *this;
        }
        matrix<T> operator*(const matrix<T>& b){
            if(yblas::ptr!=nullptr){
                return ptr->gemm(*this, b);
            }
            return *this;
        }
        matrix<T> operator*(const T b){
            if(yblas::ptr!=nullptr){
                return ptr->mul(*this, b);
            }
            return *this;
        }
        matrix<T> operator/(const T b){
            if(yblas::ptr!=nullptr){
                auto c=ptr->mul(*this, static_cast<T>(1.0)/b);
                return c;
            }
            return *this;
        }
        matrix<T> operator+(const matrix<T>& b){
            if(yblas::ptr!=nullptr){
                auto c=ptr->add(*this, b);
                return c;
            }
            return *this;
        }
        matrix<T> operator-(const matrix<T>& b){
            if(yblas::ptr!=nullptr){
                auto tmp=ptr->mul(b,-1.0);
                auto c=ptr->add(*this, tmp);
                return c;
            }
            return *this;
        }
    };
    private:
    const int _proc;
    const int _rank;
    int _icontxt;
    struct gridcom{
        int nprow;
        int npcol;
        int myrow;
        int mycol;
        int icontxt;
        gridcom(int& _icontxt, const int _proc):icontxt(_icontxt){
            nprow = std::sqrt( _proc );
            npcol = _proc / nprow;
            const int negone=-1, zero=0;
            blacs_get_( &negone, &zero, &icontxt );
            blacs_gridinit_( &icontxt, "C", &nprow, &npcol );
            blacs_gridinfo_( &icontxt, &nprow, &npcol, &myrow, &mycol );
        }
        ~gridcom(){
            blacs_gridexit_( &icontxt );
        }
    };
    void check_nblock(const int n){
        nblock=n>64?64:(n>16?16:1);
    }
    void p_gemm_impl(const matrix<double>& a_local, std::array<int, 9>& desc_a_local, matrix<double>& a_dist, std::array<int, 9>& desc_a, const matrix<double>& b_local, std::array<int, 9>& desc_b_local, matrix<double>& b_dist, std::array<int, 9>& desc_b, matrix<double>& c_local, std::array<int, 9>& desc_c_local, matrix<double>& c_dist, std::array<int, 9>& desc_c){
        const int n=a_local.ncol();
        const char trans='N';
        const double one=1, zero=0;
        const int i_one=1;
        pdgeadd_( &trans, &n, &n, &one, a_local.data().data(), &i_one, &i_one, desc_a_local.data(), &zero, a_dist.data().data(), &i_one, &i_one, desc_a.data() );
        pdgeadd_( &trans, &n, &n, &one, b_local.data().data(), &i_one, &i_one, desc_b_local.data(), &zero, b_dist.data().data(), &i_one, &i_one, desc_b.data() );
        pdgemm_( "N", "N", &n, &n, &n, &one, a_dist.data().data(), &i_one, &i_one, desc_a.data(), b_dist.data().data(), &i_one, &i_one, desc_b.data(), &zero, c_dist.data().data(), &i_one, &i_one, desc_c.data() );
        blacs_barrier_(&_icontxt,"A");
        pdgemr2d_(&n , &n , c_dist.data().data() , &i_one , &i_one , desc_c.data() , c_local.data().data() ,&i_one , &i_one , desc_c_local.data() , &_icontxt );
    }
    void p_gemm_impl(const matrix<float>& a_local, std::array<int, 9>& desc_a_local, matrix<float>& a_dist, std::array<int, 9>& desc_a, const matrix<float>& b_local, std::array<int, 9>& desc_b_local, matrix<float>& b_dist, std::array<int, 9>& desc_b, matrix<float>& c_local, std::array<int, 9>& desc_c_local, matrix<float>& c_dist, std::array<int, 9>& desc_c){
        const int n=a_local.ncol();
        const char trans='N';
        const float one=1, zero=0;
        const int i_one=1;
        psgeadd_( &trans, &n, &n, &one, a_local.data().data(), &i_one, &i_one, desc_a_local.data(), &zero, a_dist.data().data(), &i_one, &i_one, desc_a.data() );
        psgeadd_( &trans, &n, &n, &one, b_local.data().data(), &i_one, &i_one, desc_b_local.data(), &zero, b_dist.data().data(), &i_one, &i_one, desc_b.data() );
        psgemm_( "N", "N", &n, &n, &n, &one, a_dist.data().data(), &i_one, &i_one, desc_a.data(), b_dist.data().data(), &i_one, &i_one, desc_b.data(), &zero, c_dist.data().data(), &i_one, &i_one, desc_c.data() );
        blacs_barrier_(&_icontxt,"A");
        psgemr2d_(&n , &n , c_dist.data().data() , &i_one , &i_one , desc_c.data() , c_local.data().data() ,&i_one , &i_one , desc_c_local.data() , &_icontxt );
    }
    void p_gesv_impl(const matrix<double>& a_local, std::array<int, 9>& desc_a_local, matrix<double>& a_dist, std::array<int, 9>& desc_a, const matrix<double>& b_local, std::array<int, 9>& desc_b_local, matrix<double>& b_dist, std::array<int, 9>& desc_b, matrix<double>& x_local){
        const int n=a_local.ncol();
        const char trans='N';
        const double one=1, zero=0;
        const int i_one=1;
        int info;
        pdgeadd_( &trans, &n, &n, &one, a_local.data().data(), &i_one, &i_one, desc_a_local.data(), &zero, a_dist.data().data(), &i_one, &i_one, desc_a.data() );
        pdgeadd_( &trans, &n, &i_one, &one, b_local.data().data(), &i_one, &i_one, desc_b_local.data(), &zero, b_dist.data().data(), &i_one, &i_one, desc_b.data() );
        std::vector<int> ipiv(n+nblock);
        pdgesv_(&n, &i_one, a_dist.data().data(), &i_one, &i_one, desc_a.data(), ipiv.data(), b_dist.data().data(), &i_one, &i_one, desc_b.data(), &info);
        blacs_barrier_(&_icontxt,"A");
        pdcopy_(&n, b_dist.data().data(), &i_one, &i_one, desc_b.data(), &i_one, x_local.data().data(), &i_one, &i_one, desc_b_local.data(), &i_one);
    }
    void p_gesv_impl(const matrix<float>& a_local, std::array<int, 9>& desc_a_local, matrix<float>& a_dist, std::array<int, 9>& desc_a, const matrix<float>& b_local, std::array<int, 9>& desc_b_local, matrix<float>& b_dist, std::array<int, 9>& desc_b, matrix<float>& x_local){
        const int n=a_local.ncol();
        const char trans='N';
        const float one=1, zero=0;
        const int i_one=1;
        int info;
        psgeadd_( &trans, &n, &n, &one, a_local.data().data(), &i_one, &i_one, desc_a_local.data(), &zero, a_dist.data().data(), &i_one, &i_one, desc_a.data() );
        psgeadd_( &trans, &n, &i_one, &one, b_local.data().data(), &i_one, &i_one, desc_b_local.data(), &zero, b_dist.data().data(), &i_one, &i_one, desc_b.data() );
        std::vector<int> ipiv(n+nblock);
        psgesv_(&n, &i_one, a_dist.data().data(), &i_one, &i_one, desc_a.data(), ipiv.data(), b_dist.data().data(), &i_one, &i_one, desc_b.data(), &info);
        blacs_barrier_(&_icontxt,"A");
        pscopy_(&n, b_dist.data().data(), &i_one, &i_one, desc_b.data(), &i_one, x_local.data().data(), &i_one, &i_one, desc_b_local.data(), &i_one);
    }
    void p_gemv_impl(const matrix<double>& a_local, std::array<int, 9>& desc_a_local, matrix<double>& a_dist, std::array<int, 9>& desc_a, const matrix<double>& b_local, std::array<int, 9>& desc_b_local, matrix<double>& b_dist, std::array<int, 9>& desc_b, matrix<double>& x_local, std::array<int, 9>& desc_x_local, matrix<double>& x_dist, std::array<int, 9>& desc_x){
        const int n=a_local.ncol();
        const char trans='N';
        const double one=1, zero=0;
        const int i_one=1;
        int info;
        pdgeadd_( &trans, &n, &n, &one, a_local.data().data(), &i_one, &i_one, desc_a_local.data(), &zero, a_dist.data().data(), &i_one, &i_one, desc_a.data() );
        pdgeadd_( &trans, &n, &i_one, &one, b_local.data().data(), &i_one, &i_one, desc_b_local.data(), &zero, b_dist.data().data(), &i_one, &i_one, desc_b.data() );
        pdgemv_( &trans, &n, &n, &one, a_dist.data().data(), &i_one, &i_one, desc_a.data(), b_dist.data().data(), &i_one, &i_one, desc_b.data(), &i_one, &zero, x_dist.data().data(), &i_one, &i_one, desc_x.data(), &i_one );
        blacs_barrier_(&_icontxt,"A");
        pdcopy_(&n, x_dist.data().data(), &i_one, &i_one, desc_x.data(), &i_one, x_local.data().data(), &i_one, &i_one, desc_x_local.data(), &i_one);
    }
    void p_gemv_impl(const matrix<float>& a_local, std::array<int, 9>& desc_a_local, matrix<float>& a_dist, std::array<int, 9>& desc_a, const matrix<float>& b_local, std::array<int, 9>& desc_b_local, matrix<float>& b_dist, std::array<int, 9>& desc_b, matrix<float>& x_local, std::array<int, 9>& desc_x_local, matrix<float>& x_dist, std::array<int, 9>& desc_x){
        const int n=a_local.ncol();
        const char trans='N';
        const float one=1, zero=0;
        const int i_one=1;
        int info;
        psgeadd_( &trans, &n, &n, &one, a_local.data().data(), &i_one, &i_one, desc_a_local.data(), &zero, a_dist.data().data(), &i_one, &i_one, desc_a.data() );
        psgeadd_( &trans, &n, &i_one, &one, b_local.data().data(), &i_one, &i_one, desc_b_local.data(), &zero, b_dist.data().data(), &i_one, &i_one, desc_b.data() );
        psgemv_( &trans, &n, &n, &one, a_dist.data().data(), &i_one, &i_one, desc_a.data(), b_dist.data().data(), &i_one, &i_one, desc_b.data(), &i_one, &zero, x_dist.data().data(), &i_one, &i_one, desc_x.data(), &i_one );
        blacs_barrier_(&_icontxt,"A");
        pscopy_(&n, x_dist.data().data(), &i_one, &i_one, desc_x.data(), &i_one, x_local.data().data(), &i_one, &i_one, desc_x_local.data(), &i_one);
    }
    void p_syev_impl(const matrix<double>& a_local, std::array<int, 9>& desc_a_local, matrix<double>& a_dist, std::array<int, 9>& desc_a, matrix<double>& w_local, matrix<double>& z_local, std::array<int, 9>& desc_z_local, matrix<double>& z_dist, std::array<int, 9>& desc_z){
        const int n=a_local.ncol();
        const char trans='N', jobz='V', uplo='U';
        const double one=1, zero=0;
        const int i_one=1;
        int info;
        std::vector<double> workspace(n*n*2);
        int wsize=-1;
        pdgeadd_( &trans, &n, &n, &one, a_local.data().data(), &i_one, &i_one, desc_a_local.data(), &zero, a_dist.data().data(), &i_one, &i_one, desc_a.data() );
        pdsyev_( &jobz, &uplo, &n, a_dist.data().data(), &i_one, &i_one, desc_a.data(), w_local.data().data(), z_dist.data().data(), &i_one, &i_one, desc_z.data(), workspace.data(), &wsize, &info);
        blacs_barrier_(&_icontxt,"A");
        wsize=static_cast<int>(workspace.at(0));
        std::vector<double> work(wsize);
        pdsyev_( &jobz, &uplo, &n, a_dist.data().data(), &i_one, &i_one, desc_a.data(), w_local.data().data(), z_dist.data().data(), &i_one, &i_one, desc_z.data(), work.data(), &wsize, &info);
        blacs_barrier_(&_icontxt,"A");

        pdgemr2d_(&n , &n , z_dist.data().data() , &i_one , &i_one , desc_z.data() , z_local.data().data() ,&i_one , &i_one , desc_z_local.data() , &_icontxt );
    }
    void p_syev_impl(const matrix<float>& a_local, std::array<int, 9>& desc_a_local, matrix<float>& a_dist, std::array<int, 9>& desc_a, matrix<float>& w_local, matrix<float>& z_local, std::array<int, 9>& desc_z_local, matrix<float>& z_dist, std::array<int, 9>& desc_z){
        const int n=a_local.ncol();
        const char trans='N', jobz='V', uplo='U';
        const float one=1, zero=0;
        const int i_one=1;
        int info;
        std::vector<float> workspace(n*n*2);
        int wsize=-1;
        psgeadd_( &trans, &n, &n, &one, a_local.data().data(), &i_one, &i_one, desc_a_local.data(), &zero, a_dist.data().data(), &i_one, &i_one, desc_a.data() );
        pssyev_( &jobz, &uplo, &n, a_dist.data().data(), &i_one, &i_one, desc_a.data(), w_local.data().data(), z_dist.data().data(), &i_one, &i_one, desc_z.data(), workspace.data(), &wsize, &info);
        blacs_barrier_(&_icontxt,"A");
        wsize=static_cast<int>(workspace.at(0))*2;
        std::vector<float> work(wsize);
        pssyev_( &jobz, &uplo, &n, a_dist.data().data(), &i_one, &i_one, desc_a.data(), w_local.data().data(), z_dist.data().data(), &i_one, &i_one, desc_z.data(), work.data(), &wsize, &info);
        blacs_barrier_(&_icontxt,"A");

        psgemr2d_(&n , &n , z_dist.data().data() , &i_one , &i_one , desc_z.data() , z_local.data().data() ,&i_one , &i_one , desc_z_local.data() , &_icontxt );
    }
    void p_geadd_impl(const matrix<float>& a_local, std::array<int, 9>& desc_a_local, matrix<float>& c_local, std::array<int, 9>& desc_c_local){
        const int n=a_local.ncol();
        const char trans='N';
        const float one=1;
        const int i_one=1;
        psgeadd_( &trans, &n, &n, &one, a_local.data().data(), &i_one, &i_one, desc_a_local.data(), &one, c_local.data().data(), &i_one, &i_one, desc_c_local.data() );
    }
    void p_geadd_impl(const matrix<double>& a_local, std::array<int, 9>& desc_a_local, matrix<double>& c_local, std::array<int, 9>& desc_c_local){
        const int n=a_local.ncol();
        const char trans='N';
        const double one=1;
        const int i_one=1;
        pdgeadd_( &trans, &n, &n, &one, a_local.data().data(), &i_one, &i_one, desc_a_local.data(), &one, c_local.data().data(), &i_one, &i_one, desc_c_local.data() );
    }
    void p_tran_impl(const matrix<double>& a_local, std::array<int, 9> desc_a_local, matrix<double>& c_local, std::array<int, 9> desc_c_local){
        const int n=a_local.ncol();
        const double one=1, zero=0;
        const int i_one=1;
        pdtran_(&n, &n, &one, a_local.data().data(), &i_one, &i_one, desc_a_local.data(), &zero, c_local.data().data(), &i_one, &i_one, desc_c_local.data() );

    }
    void p_tran_impl(const matrix<float>& a_local, std::array<int, 9> desc_a_local, matrix<float>& c_local, std::array<int, 9> desc_c_local){
        const int n=a_local.ncol();
        const float one=1, zero=0;
        const int i_one=1;
        pstran_(&n, &n, &one, a_local.data().data(), &i_one, &i_one, desc_a_local.data(), &zero, c_local.data().data(), &i_one, &i_one, desc_c_local.data() );

    }
    double p_lange_impl(const char norm, const matrix<double>& a_local, std::array<int, 9> desc_a_local, std::vector<double>& work){
        const int n=a_local.ncol();
        const int i_one=1;
        return pdlange (&norm ,&n, &n, a_local.data().data(), &i_one, &i_one, desc_a_local.data() , work.data() );
    }
    float p_lange_impl(const char norm, const matrix<float>& a_local, std::array<int, 9> desc_a_local, std::vector<float>& work){
        const int n=a_local.ncol();
        const int i_one=1;
        return pslange (&norm ,&n, &n, a_local.data().data(), &i_one, &i_one, desc_a_local.data() , work.data() );
    }
    template<int N>
    void gebs2d_impl(std::array<int, N>& data){
        const int size=data.size();
        const int one=1;
        igebs2d_( &_icontxt, "All", " ", &size, &one, data.data(), &one );
    }
    template<int N>
    void gebr2d_impl(std::array<int, N>& data){
        const int size=data.size();
        const int zero=0, one=1;
        igebr2d_( &_icontxt, "All", " ", &size, &one, data.data(), &one, &zero, &zero );
    }
    template<int N>
    void gebs2d_impl(std::array<double, N>& data){
        const int size=data.size();
        const int one=1;
        dgebs2d_( &_icontxt, "All", " ", &size, &one, data.data(), &one );
    }
    template<int N>
    void gebr2d_impl(std::array<double, N>& data){
        const int size=data.size();
        const int zero=0, one=1;
        dgebr2d_( &_icontxt, "All", " ", &size, &one, data.data(), &one, &zero, &zero );
    }
    template<int N>
    void gebs2d_impl(std::array<float, N>& data){
        const int size=data.size();
        const int one=1;
        sgebs2d_( &_icontxt, "All", " ", &size, &one, data.data(), &one );
    }
    template<int N>
    void gebr2d_impl(std::array<float, N>& data){
        const int size=data.size();
        const int zero=0, one=1;
        sgebr2d_( &_icontxt, "All", " ", &size, &one, data.data(), &one, &zero, &zero );
    }
    void gebs2d_impl(std::vector<int>& data){
        const int size=data.size();
        const int one=1;
        igebs2d_( &_icontxt, "All", " ", &size, &one, data.data(), &one );
    }
    void gebr2d_impl(std::vector<int>& data){
        const int size=data.size();
        const int zero=0, one=1;
        igebr2d_( &_icontxt, "All", " ", &size, &one, data.data(), &one, &zero, &zero );
    }
    void gebs2d_impl(std::vector<double>& data){
        const int size=data.size();
        const int one=1;
        dgebs2d_( &_icontxt, "All", " ", &size, &one, data.data(), &one );
    }
    void gebr2d_impl(std::vector<double>& data){
        const int size=data.size();
        const int zero=0, one=1;
        dgebr2d_( &_icontxt, "All", " ", &size, &one, data.data(), &one, &zero, &zero );
    }
    void gebs2d_impl(std::vector<float>& data){
        const int size=data.size();
        const int one=1;
        sgebs2d_( &_icontxt, "All", " ", &size, &one, data.data(), &one );
    }
    void gebr2d_impl(std::vector<float>& data){
        const int size=data.size();
        const int zero=0, one=1;
        sgebr2d_( &_icontxt, "All", " ", &size, &one, data.data(), &one, &zero, &zero );
    }

    public:
    yblas():_proc([]{int rank, proc;blacs_pinfo_( &rank, &proc );return proc;}()),_rank([]{int rank, proc;blacs_pinfo_( &rank, &proc );return rank;}()),nblock(4){
        ptr=this;
    }
    ~yblas(){
        int zero=0;
        blacs_exit_( &zero );
    }
    int rank(){return _rank;}
    int proc(){return _proc;}
    template<typename T, int N>
    void bcast(std::array<T, N>& data, const int from){
        const int negone=-1,zero=0,one=1;
        const int size=data.size();
        blacs_get_( &negone, &zero, &_icontxt );
        blacs_gridinit_( &_icontxt, "C", &_proc, &one );
        if(_rank==from){
            gebs2d_impl<N>(data);
        }else{
            gebr2d_impl<N>(data);
        }
        blacs_gridexit_( &_icontxt );
    }
    template<typename T>
    void bcast(std::vector<T>& data, const int from){
        const int negone=-1,zero=0,one=1;
        const int size=data.size();
        blacs_get_( &negone, &zero, &_icontxt );
        blacs_gridinit_( &_icontxt, "C", &_proc, &one );
        if(_rank==from){
            gebs2d_impl(data);
        }else{
            gebr2d_impl(data);
        }
        blacs_gridexit_( &_icontxt );
    }
    template<typename T>
    static auto make_diag(const int m, const int n){
        matrix<T> mat(m,n);
        mat.diag();
        return mat;
    }
    template<typename T>
    auto gemm(const matrix<T>& a_local, const matrix<T>& b_local){
        gridcom g(_icontxt, _proc);
        const int n=a_local.ncol();
        const int i_zero=0;
        int info;
        check_nblock(n);
        std::array<int, 9> desc_a_local, desc_b_local, desc_c_local;
        const int mp = numroc_( &n, &nblock, &g.myrow, &i_zero, &g.nprow );
        const int nq = numroc_( &n, &nblock, &g.mycol, &i_zero, &g.npcol );
        matrix<T> a_dist(mp, nq), b_dist(mp, nq), c_dist(mp, nq), c_local(n, n);
        std::array<int, 9> desc_a, desc_b, desc_c;
        const int lld_local = std::max( numroc_( &n, &n, &g.myrow, &i_zero, &g.nprow ), 1 );
        const int lld = std::max( mp, 1 );

        descinit_( desc_a_local.data(), &n, &n, &n, &n, &i_zero, &i_zero, &_icontxt, &lld_local, &info );
        descinit_( desc_b_local.data(), &n, &n, &n, &n, &i_zero, &i_zero, &_icontxt, &lld_local, &info );
        descinit_( desc_c_local.data(), &n, &n, &n, &n, &i_zero, &i_zero, &_icontxt, &lld_local, &info );
        descinit_( desc_a.data(), &n, &n, &nblock, &nblock, &i_zero, &i_zero, &_icontxt, &lld, &info );
        descinit_( desc_b.data(), &n, &n, &nblock, &nblock, &i_zero, &i_zero, &_icontxt, &lld, &info );
        descinit_( desc_c.data(), &n, &n, &nblock, &nblock, &i_zero, &i_zero, &_icontxt, &lld, &info );
        p_gemm_impl(a_local, desc_a_local, a_dist, desc_a, b_local, desc_b_local, b_dist, desc_b, c_local, desc_c_local, c_dist, desc_c);
        return c_local;
    }
    template<typename T>
    auto gesv(const matrix<T>& a_local, const matrix<T>& b_local){
        gridcom g(_icontxt, _proc);
        const int n=a_local.ncol();
        const int i_zero=0;
        int info;
        check_nblock(n);
        std::array<int, 9> desc_a_local, desc_b_local;
        const int mp = numroc_( &n, &nblock, &g.myrow, &i_zero, &g.nprow );
        const int nq = numroc_( &n, &nblock, &g.mycol, &i_zero, &g.npcol );
        matrix<T> a_dist(mp, nq), b_dist(mp, 1);
        std::array<int, 9> desc_a, desc_b, desc_x;
        const int lld_local = std::max( numroc_( &n, &n, &g.myrow, &i_zero, &g.nprow ), 1 );
        const int lld = std::max( mp, 1 );
        const int i_one=1;
        descinit_( desc_a_local.data(), &n, &n, &n, &n, &i_zero, &i_zero, &_icontxt, &lld_local, &info );
        descinit_( desc_b_local.data(), &n, &i_one, &n, &i_one, &i_zero, &i_zero, &_icontxt, &lld_local, &info );
        descinit_( desc_a.data(), &n, &n, &nblock, &nblock, &i_zero, &i_zero, &_icontxt, &lld, &info );
        descinit_( desc_b.data(), &n, &i_one, &nblock, &i_one, &i_zero, &i_zero, &_icontxt, &lld, &info );
        auto x_local=b_local;
        p_gesv_impl(a_local, desc_a_local, a_dist, desc_a, b_local, desc_b_local, b_dist, desc_b, x_local);
        return x_local;
    }
    template<typename T>
    auto gemv(const matrix<T>& a_local, const matrix<T>& b_local){
        gridcom g(_icontxt, _proc);
        const int n=a_local.ncol();
        const int i_zero=0;
        int info;
        check_nblock(n);
        std::array<int, 9> desc_a_local, desc_b_local,desc_x_local;
        const int mp = numroc_( &n, &nblock, &g.myrow, &i_zero, &g.nprow );
        const int nq = numroc_( &n, &nblock, &g.mycol, &i_zero, &g.npcol );
        matrix<T> a_dist(mp, nq), b_dist(mp, nq),x_dist(mp,1);
        std::array<int, 9> desc_a, desc_b, desc_x;
        const int lld_local = std::max( numroc_( &n, &n, &g.myrow, &i_zero, &g.nprow ), 1 );
        const int lld = std::max( mp, 1 );
        const int i_one=1;
        descinit_( desc_a_local.data(), &n, &n, &n, &n, &i_zero, &i_zero, &_icontxt, &lld_local, &info );
        descinit_( desc_b_local.data(), &n, &i_one, &n, &i_one, &i_zero, &i_zero, &_icontxt, &lld_local, &info );
        descinit_( desc_x_local.data(), &n, &i_one, &n, &i_one, &i_zero, &i_zero, &_icontxt, &lld_local, &info );
        
        descinit_( desc_a.data(), &n, &n, &nblock, &nblock, &i_zero, &i_zero, &_icontxt, &lld, &info );
        descinit_( desc_b.data(), &n, &i_one, &nblock, &i_one, &i_zero, &i_zero, &_icontxt, &lld, &info );
        descinit_( desc_x.data(), &n, &i_one, &nblock, &i_one, &i_zero, &i_zero, &_icontxt, &lld, &info );

        auto x_local=b_local;
        p_gemv_impl(a_local, desc_a_local, a_dist, desc_a, b_local, desc_b_local, b_dist, desc_b, x_local, desc_x_local, x_dist, desc_x);

        return x_local;
    }
    template<typename T>
    auto syev(const matrix<T>& a_local){
        gridcom g(_icontxt, _proc);
        const int n=a_local.ncol();
        const int i_zero=0;
        int info;
        check_nblock(n);
        std::array<int, 9> desc_a_local, desc_w_local, desc_z_local;
        const int mp = numroc_( &n, &nblock, &g.myrow, &i_zero, &g.nprow );
        const int nq = numroc_( &n, &nblock, &g.mycol, &i_zero, &g.npcol );
        matrix<T> a_dist(mp, nq), z_dist(mp, nq), z_local(n, n), w_local(n, 1);
        std::array<int, 9> desc_a, desc_z;
        const int lld_local = std::max( numroc_( &n, &n, &g.myrow, &i_zero, &g.nprow ), 1 );
        const int lld = std::max( mp, 1 );
        const int i_one=1;
        descinit_( desc_a_local.data(), &n, &n, &n, &n, &i_zero, &i_zero, &_icontxt, &lld_local, &info );
        descinit_( desc_z_local.data(), &n, &n, &n, &n, &i_zero, &i_zero, &_icontxt, &lld_local, &info );
        descinit_( desc_w_local.data(), &n, &i_one, &n, &i_one, &i_zero, &i_zero, &_icontxt, &lld_local, &info );
        
        descinit_( desc_a.data(), &n, &n, &nblock, &nblock, &i_zero, &i_zero, &_icontxt, &lld, &info );
        descinit_( desc_z.data(), &n, &n, &nblock, &nblock, &i_zero, &i_zero, &_icontxt, &lld, &info );
        p_syev_impl(a_local, desc_a_local, a_dist, desc_a, w_local, z_local, desc_z_local, z_dist, desc_z);

        return std::make_pair(w_local, z_local);
    }
    template<typename T>
    auto add(const matrix<T>& a_local, const matrix<T>& b_local){
        auto c_local=b_local;
        gridcom g(_icontxt, _proc);
        const int n=a_local.ncol();
        const int i_zero=0;
        int info;
        std::array<int, 9> desc_a_local, desc_c_local;
        const int lld_local = std::max( numroc_( &n, &n, &g.myrow, &i_zero, &g.nprow ), 1 );

        descinit_( desc_a_local.data(), &n, &n, &n, &n, &i_zero, &i_zero, &_icontxt, &lld_local, &info );
        descinit_( desc_c_local.data(), &n, &n, &n, &n, &i_zero, &i_zero, &_icontxt, &lld_local, &info );

        p_geadd_impl(a_local, desc_a_local, c_local, desc_c_local);
        return c_local;
    }
    template<typename T>
    auto sub(const matrix<T>& a_local, const matrix<T>& b_local){
        return add(a_local, mul(b_local,-1));
    }
    auto mul(const matrix<float>& x_local, const float a){
        auto out_local=x_local;
        for(auto& v:out_local.data()){
            v=v*a;
        }
        return out_local;
    }
    auto mul(const matrix<double>& x_local, const double a){
        auto out_local=x_local;
        for(auto& v:out_local.data()){
            v=v*a;
        }
        return out_local;
    }
    template<typename T>
    auto div(const matrix<T>& x_local, const T a){
        return mul(x_local, static_cast<T>(1)/a);
    }
    template<typename T>
    auto tran(const matrix<T>& a_local){
        auto c_local=a_local;
        gridcom g(_icontxt, _proc);
        const int n=a_local.ncol();
        const int i_zero=0;
        int info;
        std::array<int, 9> desc_a_local, desc_c_local;
        const int lld_local = std::max( numroc_( &n, &n, &g.myrow, &i_zero, &g.nprow ), 1 );

        descinit_( desc_a_local.data(), &n, &n, &n, &n, &i_zero, &i_zero, &_icontxt, &lld_local, &info );
        descinit_( desc_c_local.data(), &n, &n, &n, &n, &i_zero, &i_zero, &_icontxt, &lld_local, &info );
        p_tran_impl(a_local, desc_a_local, c_local, desc_c_local);
        return c_local;
    }
    template<typename T>
    T lange(const char norm, const matrix<T>& a_local){
        gridcom g(_icontxt, _proc);
        const int n=a_local.ncol();
        const int i_zero=0;
        std::array<int, 9> desc_a_local;
        int info;
        const int lld_local = std::max( numroc_( &n, &n, &g.myrow, &i_zero, &g.nprow ), 1 );
        const int mp = numroc_( &n, &nblock, &g.myrow, &i_zero, &g.nprow );
        const int nq = numroc_( &n, &nblock, &g.mycol, &i_zero, &g.npcol );
        int size=norm=='1'?nq:(norm=='I'?mp:(norm=='F'?0:1));
        std::vector<T> work(size);
        descinit_( desc_a_local.data(), &n, &n, &n, &n, &i_zero, &i_zero, &_icontxt, &lld_local, &info );

        return p_lange_impl (norm, a_local, desc_a_local, work );
    }
};
template<typename T>
yblas::matrix<T> operator*(const T b, const yblas::matrix<T> a_local){
    if(yblas::ptr!=nullptr){
        auto c=yblas::ptr->mul(a_local, b);
        return c;
    }
    return a_local;
}

yblas* yblas::ptr=nullptr;