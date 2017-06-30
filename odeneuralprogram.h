# ifndef __ODENEURALPROGRAM__H
# define __ODENEURALPROGRAM__H
# include <neuralprogram.h>
# include <QString>
# include <QLibrary>
typedef vector<double> Data;
typedef double(*DOUBLE_FUNCTION)();
typedef int(*INTEGER_FUNCTION)();
/*	ODE1:	ODE of first order, with the boundary condition y(x0)=f0.
 *	ODE2:	ODE of second order, with the boundary conditions y(x0)=f0
 *		and y'(x0)=ff0.
 *	ODE3:	ODE of second order, with the boundary conditions y(x0)=f0
 *		and y(x1)=f1.
 * */
# define ODE1	1
# define ODE2	2
# define ODE3	3
/*	GODE1FF:	Type definition for functions of first order equations in
 *			the form GODE1FF(x,y,y')=0.
 *	GODE2FF:	Type definition for functions of second order equations in
 *			the form GODE2FF(x,y,y',y'')=0.
 *	GFODE1FF:	Same as GODE1FF, but for fortran.
 *	GFODE2FF:	Same as GODE2FF, but for fortran.
 * */
typedef double(*GODE1FF)(double,double,double);
typedef double(*GODE2FF)(double,double,double,double);
typedef double(*GFODE1FF)(double*,double*,double*);
typedef double(*GFODE2FF)(double*,double*,double*,double*);
typedef double(*GDODE1FF)(double,double,double,double,double);
typedef double(*GDODE2FF)(double,double,double,double,double,double,double);
typedef double(*GFDODE1FF)(double*,double*,double*,double*,double*);
typedef double(*GFDODE2FF)(double*,double*,double*,double*,double *,double *);
class OdeNeuralProgram :
	public NeuralProgram
{
	private:
		vector<Data> trainx;
		vector<Data> testx;
		vector<Data> valx;
        GODE1FF ode1ff;
        GODE2FF ode2ff;
        GDODE1FF dode1ff;
        GDODE2FF dode2ff;
        GFDODE1FF fdode1ff;
        GFDODE2FF fdode2ff;
        GFODE1FF fode1ff;
        GFODE2FF fode2ff;
        int 	kind;
        int	npoints;
        double 	x0,x1;
        double 	f0,f1,ff0;
        QLibrary *ptr;
	public:
        OdeNeuralProgram(QString filename);
		virtual double	getTrainError();
		virtual double	getTestError();
		virtual void    getDeriv(Data &g);
		virtual double 	penalty1();
		virtual double 	penalty2();
		virtual double 	penalty3();
		virtual double 	penalty4();
        int	getKind()	const;
        double	getX0()		const;
        double	getX1()		const;
        double	getF0()		const;
        double	getF1()		const;
        double	getFF0()	const;
        virtual void    printOutput(QString filename);
        ~OdeNeuralProgram();
};

# endif
