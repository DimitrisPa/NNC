QT += core
QT -= gui

CONFIG += c++11

TARGET = NNC
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    tolmin.cc \
    symbol.cc \
    sigprogram.cc \
    rule.cc \
    program.cc \
    population.cc \
    odeneuralprogram.cc \
    nncneuralprogram.cc \
    neuralprogram.cc \
    neuralparser.cc \
    get_options.cc \
    fparser.cc \
    doublestack.cc \
    cprogram.cc \
    converter.cc \
    pdeneuralprogram.cc \
    sodeneuralprogram.cc \
    kdvneuralprogram.cpp

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

HEADERS += \
    tolmin.h \
    symbol.h \
    sigprogram.h \
    rule.h \
    program.h \
    population.h \
    odeneuralprogram.h \
    nncneuralprogram.h \
    neuralprogram.h \
    neuralparser.h \
    get_options.h \
    fparser.hh \
    doublestack.h \
    cprogram.h \
    converter.h \
    pdeneuralprogram.h \
    sodeneuralprogram.h \
    kdvneuralprogram.h
