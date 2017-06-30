# include <pdeneuralprogram.h>
# include <math.h>
# include <QFile>
# include <QTextStream>
# include <QIODevice>
# define	LAMBDA	100.0
typedef double(*DOUBLE_FUNCTION)();
typedef int(*INTEGER_FUNCTION)();


PdeNeuralProgram::PdeNeuralProgram(QString filename)
	:NeuralProgram(2)
{
    ptr=new QLibrary(filename);
    if(ptr==NULL)
    {
    x0 = 0.0;
    x1 = 1.0;
    y0 = 0.0;
    y1 = 1.0;
    npoints = 25;
    bpoints = 50;
    f0=NULL;
    f1=NULL;
    g0=NULL;
    g1=NULL;
    ff0=NULL;
    ff1=NULL;
    fg0=NULL;
    fg1=NULL;
    pde=NULL;
    fpde=NULL;
    ptr=NULL;
    }
    else
    {
        INTEGER_FUNCTION NPOINTS, BPOINTS;
        DOUBLE_FUNCTION X0, X1, Y0, Y1;

        NPOINTS=(INTEGER_FUNCTION)ptr->resolve("getnpoints");
        if(NPOINTS==NULL)
            NPOINTS=(INTEGER_FUNCTION)ptr->resolve("_getnpoints");
        if(NPOINTS==NULL)
            NPOINTS=(INTEGER_FUNCTION)ptr->resolve("getnpoints_");
        if(NPOINTS==NULL)
            NPOINTS=(INTEGER_FUNCTION)ptr->resolve("_getnpoints_");
        if(NPOINTS==NULL) npoints=25; else npoints=NPOINTS();

        BPOINTS=(INTEGER_FUNCTION)ptr->resolve("getbpoints");
        if(BPOINTS==NULL)
            BPOINTS=(INTEGER_FUNCTION)ptr->resolve("_getbpoints");
        if(BPOINTS==NULL)
            BPOINTS=(INTEGER_FUNCTION)ptr->resolve("getbpoints_");
        if(BPOINTS==NULL)
            BPOINTS=(INTEGER_FUNCTION)ptr->resolve("_getbpoints_");
        if(BPOINTS==NULL) bpoints=50; else bpoints=BPOINTS();

        X0=(DOUBLE_FUNCTION)ptr->resolve("getx0");
        if(X0==NULL) X0=(DOUBLE_FUNCTION)ptr->resolve("_getx0");
        if(X0==NULL) X0=(DOUBLE_FUNCTION)ptr->resolve("getx0_");
        if(X0==NULL) X0=(DOUBLE_FUNCTION)ptr->resolve("_getx0_");
        if(X0==NULL) x0=0.0; else x0=X0();

        X1=(DOUBLE_FUNCTION)ptr->resolve("getx1");
        if(X1==NULL) X1=(DOUBLE_FUNCTION)ptr->resolve("_getx1");
        if(X1==NULL) X1=(DOUBLE_FUNCTION)ptr->resolve("getx1_");
        if(X1==NULL) X1=(DOUBLE_FUNCTION)ptr->resolve("_getx1_");
        if(X1==NULL) x1=1.0; else x1=X1();

        dpde=(DPDE)ptr->resolve("dpde");
        if(dpde==NULL) dpde=(DPDE)ptr->resolve("_dpde");
        if(dpde==NULL) fdpde=(FDPDE)ptr->resolve("dpde_");
        if(fdpde==NULL) fdpde=(FDPDE)ptr->resolve("_dpde_");

        ff0=NULL;
        ff1=NULL;
        fg0=NULL;
        fg1=NULL;
        fpde=NULL;

        Y0=(DOUBLE_FUNCTION)ptr->resolve("gety0");
        if(Y0==NULL) Y0=(DOUBLE_FUNCTION)ptr->resolve("_gety0");
        if(Y0==NULL) Y0=(DOUBLE_FUNCTION)ptr->resolve("gety0_");
        if(Y0==NULL) Y0=(DOUBLE_FUNCTION)ptr->resolve("_gety0_");
        if(Y0==NULL) y0=0.0; else y0=Y0();

        Y1=(DOUBLE_FUNCTION)ptr->resolve("gety1");
        if(Y1==NULL) Y1=(DOUBLE_FUNCTION)ptr->resolve("_gety1");
        if(Y1==NULL) Y1=(DOUBLE_FUNCTION)ptr->resolve("gety1_");
        if(Y1==NULL) Y1=(DOUBLE_FUNCTION)ptr->resolve("_gety1_");
        if(Y1==NULL) y1=0.0; else y1=Y1();

        f0=(GBOUNDF)ptr->resolve("f0");
        if(f0==NULL) ff0=(GFBOUNDF)ptr->resolve("_f0");
        if(f0==NULL) ff0=(GFBOUNDF)ptr->resolve("f0_");
        if(ff0==NULL) ff0=(GFBOUNDF)ptr->resolve("_f0_");

        f1=(GBOUNDF)ptr->resolve("f1");
        if(f1==NULL) f1=(GBOUNDF)ptr->resolve("_f1");
        if(f1==NULL) ff1=(GFBOUNDF)ptr->resolve("f1_");
        if(ff1==NULL) ff1=(GFBOUNDF)ptr->resolve("_f1_");

        g0=(GBOUNDF)ptr->resolve("g0");
        if(g0==NULL) g0=(GBOUNDF)ptr->resolve("_g0");
        if(g0==NULL) fg0=(GFBOUNDF)ptr->resolve("g0_");
        if(fg0==NULL) fg0=(GFBOUNDF)ptr->resolve("_g0_");

        g1=(GBOUNDF)ptr->resolve("g1");
        if(g1==NULL) g1=(GBOUNDF)ptr->resolve("_g1");
        if(g1==NULL) fg1=(GFBOUNDF)ptr->resolve("g1_");
        if(fg1==NULL) fg1=(GFBOUNDF)ptr->resolve("_g1_");

        pde=(GPDE)ptr->resolve("pde");
        if(pde==NULL) pde=(GPDE)ptr->resolve("_pde");
        if(pde==NULL) fpde=(GFPDE)ptr->resolve("pde_");
        if(fpde==NULL) fpde=(GFPDE)ptr->resolve("_pde_");
    }
    double stepx=(getX1()-getX0())/npoints;
    double stepy=(getY1()-getY0())/npoints;
    for(double x=getX0();x<=getX1();x+=stepx)
	{
        for(double y=getY0();y<=getY1();y+=stepy)
		{
			Data point;
			point.resize(2);
			point[0]=x;
			point[1]=y;
			trainx.push_back(point);
            point[0]=getX0()+(getX1()-getX0())*drand48();
            point[1]=getY0()+(getY1()-getY0())*drand48();
			valx.push_back(point);
		}
	}
    stepx=(getX1()-getX0())/(10*npoints);
    stepy=(getY1()-getY0())/(10*npoints);
    for(double x=getX0();x<=getX1();x+=stepx)
	{
        for(double y=getY0();y<=getY1();y+=stepy)
		{
			Data point;
			point.resize(2);
			point[0]=x;
			point[1]=y;
			testx.push_back(point);
		}
	}
}

double	PdeNeuralProgram::penalty1()
{
	Data point;
	point.resize(2);
    double stepx=(getX1()-getX0())/bpoints;
	double v=0.0;
    for(double x=getX0();x<=getX1();x+=stepx)
	{
		point[0]=x;
        point[1]=getY0();
		v+=pow(neuralparser->eval(point)-g0(point[0]),2.0);
	}
	return v;
}
double	PdeNeuralProgram::getX0() const
{
    return x0;
}
double  PdeNeuralProgram::getX1() const
{
    return x1;
}
double  PdeNeuralProgram::getY0() const
{
    return y0;
}
double	PdeNeuralProgram::getY1() const
{
    return y1;
}
double	PdeNeuralProgram::penalty2()
{
	Data point;
	point.resize(2);
    double stepx=(getX1()-getX0())/bpoints;
	double v=0.0;
    for(double x=getX0();x<=getX1();x+=stepx)
	{
		point[0]=x;
        point[1]=getY1();
		v+=pow(neuralparser->eval(point)-g1(point[0]),2.0);
	}
	return v;
}

double	PdeNeuralProgram::penalty3()
{
	Data point;
	point.resize(2);
    double stepy=(getY1()-getY0())/bpoints;
	double v=0.0;
    for(double y=getY0();y<=getY1();y+=stepy)
	{
        point[0]=getX0();
		point[1]=y;
		v+=pow(neuralparser->eval(point)-f0(point[1]),2.0);
	}

	return v;
}

double	PdeNeuralProgram::penalty4()
{
	Data point;
	point.resize(2);
    double stepy=(getY1()-getY0())/bpoints;
	double v=0.0;
    for(double y=getY0();y<=getY1();y+=stepy)
	{
        point[0]=getX1();
		point[1]=y;
		v+=pow(neuralparser->eval(point)-f1(point[1]),2.0);
	}
	return v;
}

void	PdeNeuralProgram::getDeriv(Data &g)
{

	for(int i=0;i<g.size();i++) g[i]=0.0;
	Data tempv,tempx1,tempx2,tempy1,tempy2;
	tempv.resize(g.size());
	tempx1.resize(g.size());
	tempx2.resize(g.size());
	tempy1.resize(g.size());
	tempy2.resize(g.size());
	for(int i=0;i<trainx.size();i++)
	{
		double x=trainx[i][0];
		double y=trainx[i][1];
		double v=neuralparser->eval(trainx[i]);
		double x1=neuralparser->evalDeriv(trainx[i],1);
		double y1=neuralparser->evalDeriv(trainx[i],2);
		double x2=neuralparser->evalDeriv2(trainx[i],1);
		double y2=neuralparser->evalDeriv2(trainx[i],2);
		double gg=pde(x,y,v,x1,y1,x2,y2);
		neuralparser->getDeriv(trainx[i],tempv);
		neuralparser->getXDeriv(trainx[i],1,tempx1);
		neuralparser->getXDeriv(trainx[i],2,tempy1);
		neuralparser->getX2Deriv(trainx[i],1,tempx2);
		neuralparser->getX2Deriv(trainx[i],2,tempy2);
		for(int j=0;j<g.size();j++)
		{
			double out;
			out=dpde(x,y,v,x1,y1,x2,y2,tempv[j],tempx1[j],tempy1[j],tempx2[j],tempy2[j]);
			g[j]+=2.0*out*gg;
		}
	}
	Data point;
	point.resize(2);
    double stepx=(getX1()-getX0())/bpoints;
    double stepy=(getY1()-getY0())/bpoints;
    for(double x=getX0();x<=getX1();x+=stepx)
	{
		point[0]=x;
        point[1]=getY0();
		neuralparser->getDeriv(point,tempv);
		double val=neuralparser->eval(point);
		for(int j=0;j<g.size();j++) g[j]+=2.0*LAMBDA*(val-g0(x))*tempv[j];
        point[1]=getY1();
		neuralparser->getDeriv(point,tempv);
		val=neuralparser->eval(point);
		for(int j=0;j<g.size();j++) g[j]+=2.0*LAMBDA*(val-g1(x))*tempv[j];
	}
    for(double y=getY0();y<=getY1();y+=stepy)
	{
        point[0]=getX0();
		point[1]=y;
		neuralparser->getDeriv(point,tempv);
		double val=neuralparser->eval(point);
		for(int j=0;j<g.size();j++) g[j]+=2.0*LAMBDA*(val-f0(y))*tempv[j];
        point[0]=getX1();
		val=neuralparser->eval(point);
		neuralparser->getDeriv(point,tempv);
		for(int j=0;j<g.size();j++) g[j]+=2.0*LAMBDA*(val-f1(y))*tempv[j];
	}
}


double	PdeNeuralProgram::getTrainError()
{
    double value=0.0;
    for(int i=0;i<trainx.size();i++)
	{
		double x=trainx[i][0];
		double y=trainx[i][1];
		double v=neuralparser->eval(trainx[i]);
		double x1=neuralparser->evalDeriv(trainx[i],1);
		double y1=neuralparser->evalDeriv(trainx[i],2);
		double x2=neuralparser->evalDeriv2(trainx[i],1);
		double y2=neuralparser->evalDeriv2(trainx[i],2);
		double g=pde(x,y,v,x1,y1,x2,y2);
		value=value+g*g;
	}
	double penalty=penalty1()+penalty2()+penalty3()+penalty4();
	return value+LAMBDA * penalty;
}

double	PdeNeuralProgram::getTestError()
{
    double value=0.0;
	for(int i=0;i<testx.size();i++)
	{
		double x=testx[i][0];
		double y=testx[i][1];
		double v=neuralparser->eval(testx[i]);
		double x1=neuralparser->evalDeriv(testx[i],1);
		double y1=neuralparser->evalDeriv(testx[i],2);
		double x2=neuralparser->evalDeriv2(testx[i],1);
		double y2=neuralparser->evalDeriv2(testx[i],2);
		double g=pde(x,y,v,x1,y1,x2,y2);
		value=value+g*g;
	}
	return value;
}
void    PdeNeuralProgram::printOutput(QString filename)
{
    QFile fp(filename);
    if(!fp.open(QIODevice::WriteOnly |QIODevice::Text)) return;
    QTextStream st(&fp);
    for(int i=0;i<testx.size();i++)
    {
        double v=neuralparser->eval(testx[i]);
        st<<testx[i][0]<<" "<<testx[i][1]<<" "<<v<<endl;
    }
    fp.close();
}
