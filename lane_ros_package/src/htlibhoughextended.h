/*
 * Copyright (c) 2013, University of Zagreb (http://www.unizg.hr/)
 * Copyright (c) 2013, Faculty of Electrical Engineering and Computing (http://www.fer.unizg.hr/)
 * Copyright (c) 2013, Tomislav Petkoviæ (tomislav.petkovic@gmail.com)
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * - Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 * - Neither the name of the University of Zagreb nor the
 *   names of its contributors may be used to endorse or promote products
 *   derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL UNIVERSITY OF ZAGREB BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*!
  \file   htlibhoughextended.h
  \brief  Implementation of the Hough transform.

  Extended Hough transform that uses accumulator that includes all possible angles.

  \author Tomislav Petkovic
  \date   2013-04-11
*/


#ifndef __HT_HTLIBHOUGHEXTENDED_H
#define __HT_HTLIBHOUGHEXTENDED_H

/*! A value of PI. */
#define HT_CONSTANT_PI 3.141592653589793238462643383279502884197169399375
/*! A value of PI (float). */
#define HT_CONSTANT_PIF 3.141592653589793238462643383279502884197169399375f
/*! A value of 1/PI (float). */
#define HT_CONSTANT_INVPIF 0.3183098861837906715377675267450287240689192914809f
/*! A value of PI half. */
#define HT_CONSTANT_PIHALF 1.570796326794896619231321691639751442098584699687
/*! A value of PI half (float). */
#define HT_CONSTANT_PIHALFF 1.570796326794896619231321691639751442098584699687f
/*! A value of inverse PI half or Buffons constant (float). */
#define HT_CONSTANT_INVPIHALFF 0.63661977236758134307553505349005744813783858296182579499f
/*! A value of PI third (float). */
#define HT_CONSTANT_PITHIRDF 1.0471975511965977461542144610931676280657231331f
/*! A value of PI quarter (float). */
#define HT_CONSTANT_PIQUARTERF 0.785398163397448309615660845819875721049292349843776455243736f
/*! A value of PI eigth (float). */
#define HT_CONSTANT_PIEIGHTF 0.5235987755982988730771072305465838140328615665625176368291574320f
/*! A value of tan of one eight of pi (float). */
#define HT_CONSTANT_TANPIEIGHTF 0.414213562373095048801688724209698078569671875376f
/*! A value of square root of 2. */
#define HT_CONSTANT_SQRT_2 1.414213562373095048801688724209698078569671875376
/*! A value of square root of 2 (float). */
#define HT_CONSTANT_SQRT_2F 1.414213562373095048801688724209698078569671875376f
/*! A value of square root of 3. */
#define HT_CONSTANT_SQRT_3 1.732050807568877293527446341505872366942805253810
/*! A value of square root of 3 (float). */
#define HT_CONSTANT_SQRT_3F 1.732050807568877293527446341505872366942805253810f
/*! A inverse value of square root of 2 (float). */
#define HT_CONSTANT_INVSQRT_2F 0.707106781186547524400844362104849039284835937688474036f
/*! A square root of 2 times natural logarithm of 2 (float). */
#define HT_CONSTANT_SQRTTWOLNTWO 1.17741002251547469101156932645969963774738568938582053852252575f

#include "htlibaux.h"
#include <set>
#include <queue>

#include <iostream>

using namespace std;
//! Finds lines using Hough transform.
/*!
  Function finds lines using Hough transform.

  This is adopted from the OpenCV 2.3.1 implementation from file
  \\modules\\imgproc\\src\\hough.cpp. Reimplementation stores
  the accumulator in the normal cv::Mat image and uses somewhat
  improved accumulation technique where local edge orientation is
  taken into account thus improving the accumulator image.
  Accumulator is also extended so edge point are differentiatied
  based on gradient orientation.

  \param img    Input image (binary). Usually a Canny edge map is used.
  \param img_dx Sobel derivative along x axis.
  \param img_dy Sobel derivative along y axis.
  \param lines  Vector of found lines.
  \param rho    Distance resolution in pixels.
  \param theta  Angle resolution in radians.
  \param acc_threshold  Minimal value of accumulator image.
  \param lines_max    Maximal number of lines to detect.
  \param accum  Line accumulator image.
  \return Function returns true if successfull.
*/
inline
bool
htHoughLinesExtended_inline(
                            cv::Mat & img,
                            cv::Mat & img_dx,
                            cv::Mat & img_dy,
                            std::vector<cv::Vec2f> & lines,
                            float const rho,
                            float const theta,
                            int const acc_threshold,
                            int const lines_max,
                            cv::Mat & accum
                            )
{
  /* Sanity check. */
  assert( (0 < theta) && (theta < HT_CONSTANT_PIEIGHTF) );
  if (0 >= theta) return( false );

  assert( 0 < rho );
  if (0 >= rho) return( false );

  assert( CV_8UC1 == img.type() );
  if ( CV_8UC1 != img.type() ) return( false );

  assert( CV_16SC1 == img_dx.type() );
  if ( CV_16SC1 != img_dx.type() ) return( false );

  assert( CV_16SC1 == img_dy.type() );
  if ( CV_16SC1 != img_dx.type() ) return( false );

  bool result = true;

  int const rows = img.rows;
  int const cols = img.cols;
  if ( (0 >= rows) || (0 >= cols) ) return( result );

  assert( (rows == img_dx.rows) && (cols == img_dx.cols) );
  if ( (rows != img_dx.rows) || (cols != img_dx.cols) ) return( false );

  assert( (rows == img_dy.rows) || (cols == img_dy.cols) );
  if ( (rows != img_dy.rows) || (cols != img_dy.cols) ) return( false );


  /* Preallocate storage. Accumulator is extended to make local extrema easier to find. */
  float const xhalf = (float)(cols) * 0.5f;
  float const yhalf = (float)(rows) * 0.5f;

  float const itheta = 1.0f / theta;
  float const irho = 1.0f / rho;

  float const numangle_f = 2 * HT_CONSTANT_PIF * itheta;
  float const numrho_f = 2 * sqrtf(xhalf * xhalf + yhalf * yhalf) * irho;

  float const ang_offset = numangle_f * 0.5f;
  float const rho_offset = numrho_f * 0.5f;

  int const numangle = inlineceil_f( numangle_f );
  int const numrho = inlineceil_f( numrho_f );

  int const ext = 2;
  accum = cv::Mat::zeros( numrho + 2 * ext, numangle + 2 * ext, CV_32SC1 );

  int const arows = accum.rows;
  int const acols = accum.cols;

  cv::AutoBuffer<int> _sort_buf;
  _sort_buf.allocate(numangle * numrho);
  int * const sort_buf = _sort_buf;


  /* Precompute sine and cosine tables. */
  cv::AutoBuffer<float> _tabSin, _tabCos;
  _tabSin.allocate(acols);
  _tabCos.allocate(acols);
  float * const tabSin = _tabSin;
  float * const tabCos = _tabCos;
  {
    float ang = - ( (float)(ext) + ang_offset ) * theta;
    for(int n = 0; n < acols; ang += theta, ++n)
      {
        tabSin[n] = sinf(ang) * irho;
        tabCos[n] = cosf(ang) * irho;
      }
    /* for */
  }


  /* Fill accumulator image. */
  int const istep0 = (int)( img.step[0] );
  uchar * const idata = img.data;

  int const dxstep0 = (int)( img_dx.step[0] );
  uchar * const dxdata = img_dx.data;

  int const dystep0 = (int)( img_dy.step[0] );
  uchar * const dydata = img_dy.data;

  int const astep0 = (int)( accum.step[0] );
  int const astep1 = (int)( accum.step[1] );
  uchar * const adata = accum.data;

  int const i_offset = inlineround_f(ang_offset) + ext;
  int const n_offset = numangle - 1;
  int const n_left = inlineceil_f( (float)(ext) + ang_offset - HT_CONSTANT_PIF * itheta );
  int const n_right = inlinefloor_f( (float)(ext) + ang_offset + HT_CONSTANT_PIF * itheta );

  for (int y = 0; y < rows; ++y)
    {
      uchar const * const idata_row = (uchar *)( idata + y * istep0 );
      short const * const dx_row = (short *)(dxdata + y * dxstep0);
      short const * const dy_row = (short *)(dydata + y * dystep0);

      for (int x = 0; x < cols; ++x)
        {
          if (0 != idata_row[x])
            {
              /* Find starting and ending angle. */
              float const vx = (float)( dx_row[x] );
              float const vy = (float)( dy_row[x] );

              float const ang = atan2f(vy, vx);

              // cout<<ang * (180/HT_CONSTANT_PI)<<'\n';

              // if (ang * (180/HT_CONSTANT_PI) > 100){

                
              float const angs = ang - HT_CONSTANT_PIEIGHTF;
              float const ange = ang + HT_CONSTANT_PIEIGHTF;

              int const start = inlineround_f(angs * itheta);
              int const end = inlineround_f(ange * itheta);

              assert(start <= end);

              /* Draw sine into accumulator image. */
              float const xv = (float)(x) - xhalf;
              float const yv = (float)(y) - yhalf;
              for (int i = start; i < end; ++i)
                {
                  /* Compute angle index. */
                  int const n = i + i_offset;

                  if ( (0 <= n) && (n < acols) )
                    {
                      /* Compute accumulator coordinates. */
                      float const value = xv * tabCos[n] + yv * tabSin[n] + rho_offset;
                      int const m = inlineround_f( value ) + ext;

                      /* Accumulate. */
                      assert( (0 <= m) && (m < arows) );
                      if ( (0 <= m) && (m < arows) )
                        {
                          int * const adata_row = (int *)( adata + m * astep0 );
                          ++( adata_row[n] );
                        }
                      /* if */
                    }
                  /* if */

                  if (n_left >= n)
                    {
                      /* Compute coordinates for wrap around from left to right. */
                      int const n1 = n + n_offset;
                      assert( (0 <= n1) && (n1 < acols) );
                      if ( (0 <= n1) && (n1 < acols) )
                        {
                          float const value = xv * tabCos[n1] + yv * tabSin[n1] + rho_offset;
                          int const m = inlineround_f( value ) + ext;
                          assert( (0 <= m) && (m < arows) );
                          if ( (0 <= m) && (m < arows) )
                            {
                              int * const adata_row = (int *)( adata + m * astep0 );
                              ++( adata_row[n1] );
                            }
                          /* if */
                        }
                      /* if */
                    }
                  /* if */
                  
                  if (n_right <= n)
                    {
                      /* Compute coordinates for wrap around from right to left. */
                      int const n2 = n - n_offset;
                      assert( (0 <= n2) && (n2 < acols) );
                      if ( (0 <= n2) && (n2 < acols) )
                        {
                          float const value = xv * tabCos[n2] + yv * tabSin[n2] + rho_offset;
                          int const m = inlineround_f( value ) + ext;
                          // assert( (0 <= m) && (m < numrho) );
                          if ( (0 <= m) && (m < numrho) )
                            {
                              int * const adata_row = (int *)( adata + m * astep0 );
                              ++( adata_row[n2] );
                            }
                          /* if */
                        }
                      /* if */
                    }
                  /* if */
                }
              /* for */
              // }
            }
          /* if */
        }
      /* for */
    }
  /* for */


  /* Find local extrema. Local extrema can be either isolated points or set of
     adjacent points having the same value. For the set of adjacent points with the
     same value we only select one representative point as the extrema.
  */
  int total = 0; // Number of local extremas.
  int const offset = 1;
  assert( (0 < offset) && (offset <= ext) );

  std::set<int> added; // List of points with same extrema value that are already processed.

  for (int y = ext; y < arows - ext; ++y)
    {
      int * const adata_row_prev = (int *)( adata + ( y - offset ) * astep0 );
      int * const adata_row =      (int *)( adata +   y            * astep0 );
      int * const adata_row_next = (int *)( adata + ( y + offset ) * astep0 );

      for (int x = ext; x < acols - ext; ++x)
        {
          int const base = y * astep0 + x * astep1;
          int const central = adata_row[x];
          if( central > acc_threshold )
            {
              if ( ( central > adata_row[x - offset] ) &&
                   ( central > adata_row[x + offset] ) &&
                   ( central > adata_row_prev[x - offset] ) &&
                   ( central > adata_row_prev[x         ] ) &&
                   ( central > adata_row_prev[x + offset] ) &&
                   ( central > adata_row_next[x - offset] ) &&
                   ( central > adata_row_next[x         ] ) &&
                   ( central > adata_row_next[x + offset] )
                   )
                {
                  sort_buf[total] = base; // Add single point local extrema.
                  ++total;
                }
              else if ( ( central >= adata_row[x - offset] ) &&
                        ( central >= adata_row[x + offset] ) &&
                        ( central >= adata_row_prev[x - offset] ) &&
                        ( central >= adata_row_prev[x         ] ) &&
                        ( central >= adata_row_prev[x + offset] ) &&
                        ( central >= adata_row_next[x - offset] ) &&
                        ( central >= adata_row_next[x         ] ) &&
                        ( central >= adata_row_next[x + offset] )
                        )
                {
                  if ( added.end() == added.find(base) )
                    {
                      std::set<int> processed;
                      std::queue<ht_hough_point> points;

                      ht_hough_point pt;
                      pt.x = x;
                      pt.y = y;
                      pt.base = base;
                      points.push(pt);

                      while ( false == points.empty() )
                        {
                          pt = points.front();
                          points.pop();

                          int const xl = pt.x - offset;                          
                          int const xc = pt.x;
                          int const xr = pt.x + offset;

                          int const yu = pt.y - offset;
                          int const yc = pt.y;
                          int const yd = pt.y + offset;

                          int * const ptr1 = (int *)( adata + yu * astep0 ) + xc;
                          int * const ptr2 = (int *)( adata + yc * astep0 ) + xc;
                          int * const ptr3 = (int *)( adata + yd * astep0 ) + xc;

                          if ( (central < ptr1[-offset]) || (central < ptr1[0]) || (central < ptr1[offset]) ||
                               (central < ptr2[-offset]) ||                        (central < ptr2[offset]) ||
                               (central < ptr3[-offset]) || (central < ptr3[0]) || (central < ptr3[offset])
                               )
                            {
                              processed.clear(); // Terminate if set is not local extrema.
                              break;
                            }
                          else
                            {                      
                              /* Add currrent central point and neighbours if not already added. */
                              processed.insert(pt.base);

                              /* Points one row up. */
                              pt.x = xl;
                              pt.y = yu;
                              pt.base = pt.y * astep0 + pt.x * astep1;
                              if ( (central == ((int *)(adata + pt.base))[0]) && (processed.end() == processed.find(pt.base)) &&
                                   (ext <= pt.x) && (pt.x < acols - ext) &&
                                   (ext <= pt.y) && (pt.y < arows - ext)
                                   )
                                {
                                  points.push(pt); 
                                }
                              /* if */

                              pt.x = xc;
                              pt.y = yu;
                              pt.base = pt.y * astep0 + pt.x * astep1;
                              if ( (central == ((int *)(adata + pt.base))[0]) && (processed.end() == processed.find(pt.base)) &&
                                   (ext <= pt.x) && (pt.x < acols - ext) &&
                                   (ext <= pt.y) && (pt.y < arows - ext)
                                   )
                                {
                                  points.push(pt); 
                                }
                              /* if */

                              pt.x = xr;
                              pt.y = yu;
                              pt.base = pt.y * astep0 + pt.x * astep1;
                              if ( (central == ((int *)(adata + pt.base))[0]) && (processed.end() == processed.find(pt.base)) &&
                                   (ext <= pt.x) && (pt.x < acols - ext) &&
                                   (ext <= pt.y) && (pt.y < arows - ext)
                                   )
                                {
                                  points.push(pt); 
                                }
                              /* if */

                              /* Points on the same row. */
                              pt.x = xl;
                              pt.y = yc;
                              pt.base = pt.y * astep0 + pt.x * astep1;
                              if ( (central == ((int *)(adata + pt.base))[0]) && (processed.end() == processed.find(pt.base)) &&
                                   (ext <= pt.x) && (pt.x < acols - ext) &&
                                   (ext <= pt.y) && (pt.y < arows - ext)
                                   )
                                {
                                  points.push(pt); 
                                }
                              /* if */

                              pt.x = xr;
                              pt.y = yc;
                              pt.base = pt.y * astep0 + pt.x * astep1;
                              if ( (central == ((int *)(adata + pt.base))[0]) && (processed.end() == processed.find(pt.base)) &&
                                   (ext <= pt.x) && (pt.x < acols - ext) &&
                                   (ext <= pt.y) && (pt.y < arows - ext)
                                   )
                                {
                                  points.push(pt); 
                                }
                              /* if */

                              /* Point one row down. */
                              pt.x = xl;
                              pt.y = yd;
                              pt.base = pt.y * astep0 + pt.x * astep1;
                              if ( (central == ((int *)(adata + pt.base))[0]) && (processed.end() == processed.find(pt.base)) &&
                                   (ext <= pt.x) && (pt.x < acols - ext) &&
                                   (ext <= pt.y) && (pt.y < arows - ext)
                                   )
                                {
                                  points.push(pt); 
                                }
                              /* if */

                              pt.x = xc;
                              pt.y = yd;
                              pt.base = pt.y * astep0 + pt.x * astep1;
                              if ( (central == ((int *)(adata + pt.base))[0]) && (processed.end() == processed.find(pt.base)) &&
                                   (ext <= pt.x) && (pt.x < acols - ext) &&
                                   (ext <= pt.y) && (pt.y < arows - ext)
                                   )
                                {
                                  points.push(pt); 
                                }
                              /* if */

                              pt.x = xr;
                              pt.y = yd;
                              pt.base = pt.y * astep0 + pt.x * astep1;
                              if ( (central == ((int *)(adata + pt.base))[0]) && (processed.end() == processed.find(pt.base)) &&
                                   (ext <= pt.x) && (pt.x < acols - ext) &&
                                   (ext <= pt.y) && (pt.y < arows - ext)
                                   )
                                {
                                  points.push(pt); 
                                }
                              /* if */
                            }
                          /* if */
                        }
                      /* while */

                      /* If there are points add one representative. */
                      if ( false == processed.empty() )
                        {
                          int xsum = 0;
                          int ysum = 0;
                          int n = 0;
                          std::set<int>::iterator it;

                          for (it = processed.begin(); processed.end() != it; ++it)
                            {
                              int const ib = *it;
                              if (ib > base) added.insert(ib);

                              int const iy = ib / astep0;
                              int const ix = (ib - iy * astep0) / astep1;
                              assert(ib == iy * astep0 + ix * astep1);

                              xsum += ix;
                              ysum += iy;
                              ++n;
                            }
                          /* for */

                          assert(0 < n);

                          float const in = 1.0f / (float)( n );
                          float const xadd = (float)(xsum) * in;
                          float const yadd = (float)(ysum) * in;
                          
                          int xmin = 0;
                          int ymin = 0;
                          int badd = -1;
                          float dmin = FLT_MAX;

                          for (it = processed.begin(); processed.end() != it; ++it)
                            {
                              int const ib = *it;

                              int const iy = ib / astep0;
                              int const ix = (ib - iy * astep0) / astep1;
                              assert(ib == iy * astep0 + ix * astep1);

                              float const dx = (float)(ix) - xadd;
                              float const dy = (float)(iy) - yadd;
                              float const d2 = dx * dx + dy * dy;
                              if (d2 < dmin)
                                {
                                  dmin = d2;
                                  xmin = ix;
                                  ymin = iy;
                                  badd = ib;
                                }
                              /* */
                            }
                          /* for */

                          assert( processed.end() != processed.find(badd) );

                          sort_buf[total] = badd; // Add multi-point local extrema.
                          ++total;
                          ((int *)(adata + badd))[0] = central + (int)(processed.size());
                        }
                      /* if */
                    }
                  /* if */
                }              
              /* if */
            }
          /* if */
        }
      /* for */
    }
  /* for */


  /* Sort local extrema in descending order. */
  htHoughLinesSortDescentAligned32s( sort_buf, total, adata );

  /* Store first lines_max lines that are local extrema. */
  int const lmax = (lines_max < total)? lines_max : total;
  for (int i = 0; i < lmax; ++i)
    {
      int const base = sort_buf[i];
      int const y = base / astep0;
      int const x = (base - y * astep0) / astep1;
      assert(base == y * astep0 + x * astep1);

      float const angle = ( (float)(x - ext) - ang_offset ) * theta;
      float const distance = ( (float)(y - ext) - rho_offset ) * rho;

      /* Angle and distance are for coordinate system centered in the image.
         To conform to OpenCV convention we shift the distance and move origin to upper left corner
      */
      cv::Vec2f data;
      data[0] = distance + xhalf * cosf(angle) + yhalf * sinf(angle);
      data[1] = angle;

      /* If needed we can also correct for negative distances. */
      //if (data[0] < 0)
      //  {
      //    data[0] = -data[0];
      //    data[1] += HT_CONSTANT_PIF;
      //  }
      ///* if */

      lines.push_back(data);
    }
  /* for */

  return( result );
}
/* htHoughLinesExtended_inline */



#endif /* !__HT_HTLIBHOUGHEXTENDED_H */