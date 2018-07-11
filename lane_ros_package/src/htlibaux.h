/*
 * Copyright (c) 2012, University of Zagreb (http://www.unizg.hr/)
 * Copyright (c) 2012, Faculty of Electrical Engineering and Computing (http://www.fer.unizg.hr/)
 * Copyright (c) 2012, Tomislav Petkoviæ (mailto:tomislav.petkovic@gmail.com)
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
  \file   htlibaux.h
  \brief  Helper functions.

  Auxiliary and small helper functions for the HT library.

  \author Tomislav Petkovic
  \date   2012-11-20
*/


#ifndef __HT_HTLIBAUX_H
#define __HT_HTLIBAUX_H


#include <opencv2/core/internal.hpp>

#define hough_lines_aligned_cmp_gt(l1,l2) ( ( (int *)(aux + (l1)) )[0] > ( (int *)(aux + (l2)) )[0] )

static CV_IMPLEMENT_QSORT_EX( htHoughLinesSortDescentAligned32s, int, hough_lines_aligned_cmp_gt, uchar const * );

//! Point storage.
/*!
  When finding extrema of the Hough accumulators there can be sets of connected points having same
  acuumulator value. If those are local extrema then one representative point must be added to the
  accumulator. This structure is used to track data of such points.
*/
typedef struct _ht_hough_point
{
  int x; /*!< X coordinate. */
  int y; /*!< Y coordinate. */
  int base; /*!< Base address. */
} ht_hough_point;



//! Rounding function.
/*!
  Use for rounding a float to the nearest integer.

  \param x    Input value.
  \return Function returns nearest integer.
*/
inline
int
inlineround_f(
              float const x
              )
{
  if (x < 0)
    {
      assert(x > INT_MIN);
      return (int)(x - 0.5f);
    }
  else
    {
      assert(x < INT_MAX);
      return (int)(x + 0.5f);
    }
  /* if */
}
/* inlineround_f */



//! Rounding function.
/*!
  Use for rounding a float to the nearest lower integer.

  \param x    Input value.
  \return Function returns nearest integer.
*/
inline
int
inlinefloor_f(
              float const x
              )
{
  return (int)( floorf(x) );
}
/* inlinefloor_f */



//! Rounding function.
/*!
  Use for rounding a float to the nearest lower integer.

  \param x    Input value.
  \return Function returns nearest integer.
*/
inline
int
inlineceil_f(
              float const x
              )
{
  return (int)( ceilf(x) );
}
/* inlineceil_f */



//! Rounding function.
/*!
  Use for rounding a float to the nearest integer.

  \param x    Input value.
  \return Function returns nearest integer.
*/
inline
int
inlineround_d(
              double const x
              )
{
  if (x < 0)
    {
      assert(x > INT_MIN);
      return (int)(x - 0.5);
    }
  else
    {
      assert(x < INT_MAX);
      return (int)(x + 0.5);
    }
  /* if */
}
/* inlineround_d */



#endif /* !__HT_HTLIBAUX_H */