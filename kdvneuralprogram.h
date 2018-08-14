#ifndef KDVNEURALPROGRAM_H
#define KDVNEURALPROGRAM_H

# include <neuralprogram.h>
# include <QLibrary>
typedef vector<double> Data;
typedef double(*GBOUNDF)(double);
typedef double(*GFBOUNDF)(double*);
typedef double(*KDV)(double,double,double,double,double,double,double,double,double);
typedef double(*PDV)(double,double,double,double,double,double,double,double,double,
                     double,double,double,double,double,double,double);
class KdvNeuralProgram :public NeuralProgram
{
private:
    vector<Data> trainx;
    vector<Data> testx;
    vector<Data> valx;
    QLibrary *ptr;
    GBOUNDF f0, f1, g0, g1;
    GFBOUNDF ff0,ff1,fg0,fg1;
    KDV kdv;
    PDV pdv;
    double x0,x1,y0,y1;
    int npoints, bpoints;
public:
    KdvNeuralProgram(QString filename);
    virtual void	getDeriv(Data &g);
    virtual double	getTrainError();
    virtual double	getTestError();
    virtual double 	penalty1();
    virtual double 	penalty2();
    virtual double 	penalty3();
    virtual double 	penalty4();
    virtual void    printOutput(QString filename);
    double	getX0() const;
    double  getX1() const;
    double  getY0() const;
    double	getY1() const;
};

#endif // KDVNEURALPROGRAM_H
