/*******************************************************************************************************************
 * FILE NAME   :    all_common.h
 *
 * PROJECTION  :    general c++ lib for video processing
 *
 * DESCRIPTION :    common included headers
 *
 * VERSION HISTORY
 * YYYY/MMM/DD      Author          Comments
 * 2020 JUL 08      Yu Liu          Creation
 *
 ********************************************************************************************************************/
#pragma once

// libraries
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include <cerrno>
#include <cassert>
#include <cinttypes>

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <complex>
#include <vector>
#include <string>
#include <random>
#include <atomic>
#include <queue>
#include <map>

// defines
#define F_PI    3.14159265359f
#define F_2PI   (2.f * F_PI)
#define F_3PI   (3.f * F_PI)
#define F_4PI   (4.f * F_PI)
#define F_PI2   (F_PI / 2.f)
#define F_PI3   (F_PI / 3.f)
#define F_PI4   (F_PI / 4.f)

#ifndef M_PI
    #define M_PI        3.14159265358979323846   // pi
    #define M_2PI      (2.*M_PI)
    #define M_PI2       1.57079632679489661923   // pi/2
    #define M_PI4       0.785398163397448309616  // pi/4
    #define M_1_PI      0.318309886183790671538  // 1/pi
    #define M_2_PI      0.636619772367581343076  // 2/pi
    #define M_2_SQRTPI  1.12837916709551257390   // 2/sqrt(pi)
    #define M_SQRT2     1.41421356237309504880   // sqrt(2)
    #define M_SQRT1_2   0.707106781186547524401  // 1/sqrt(2)
#endif

typedef unsigned char   uchar;
typedef unsigned short  ushort;
typedef unsigned int    uint;

// Global Variables
extern int param_KeyLevel;
extern int param_KeyScale;
extern int default_scale;
extern int button_id;