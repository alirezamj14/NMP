import matplotlib.pyplot as plt 
import os
import joblib
import numpy as np
from scipy.io import loadmat
import time
import pickle
import argparse
from sklearn.pipeline import make_pipeline
import random
import pandas as pd
from MyFunctions import *

# Features from deeplift
S_hat = np.array([318, 403, 290, 375, 210, 513, 317, 346, 402, 345, 570, 514, 212,
       457, 404, 263, 374, 376, 401, 319, 373, 291, 400, 208, 430, 433,
       429, 485, 458, 265, 328, 432, 238, 211, 184, 322, 405, 240, 321,
       213, 209, 569, 183, 599, 598, 264, 378, 541, 239, 408, 407, 545,
       382, 350, 428, 431, 347, 292, 546, 410, 597, 266, 379, 438, 406,
       268, 214, 512, 601, 320, 236, 380, 316, 235, 434, 154, 439, 409,
       542, 352, 207, 157, 486, 628, 155, 572, 605, 186, 600, 627, 242,
       577, 241, 578, 573, 234, 606, 596, 156, 549, 490, 353, 381, 515,
       466, 576, 492, 574, 262, 518, 327, 296, 626, 461, 625, 323, 315,
       351, 285, 462, 256, 383, 489, 181, 571, 543, 289, 182, 456, 185,
       297, 540, 484, 602, 295, 237, 377, 460, 437, 520, 267, 325, 657,
       516, 517, 270, 521, 519, 436, 656, 158, 411, 126, 269, 487, 547,
       372, 206, 294, 629, 300, 326, 658, 355, 465, 459, 153, 180, 288,
       550, 233, 243, 464, 271, 272, 286, 261, 468, 488, 435, 298, 354,
       287, 324, 349, 491, 344, 356, 568, 659, 544, 604, 522, 493, 205,
       567, 630, 463, 632, 603, 293, 595, 348, 575, 440, 399, 655, 152,
       496, 523, 231, 467, 232, 548, 539, 495, 624, 494, 299, 187, 631,
       633, 204, 230, 215, 455, 244, 179, 551, 130, 634, 314, 427, 524,
       579, 329, 260, 343, 127, 159, 357, 607, 371, 386, 412, 441, 188,
       511, 684, 483, 203, 216, 257, 129, 414, 660, 125, 413, 662, 654,
       686, 151, 623, 683, 384, 160, 398, 178, 482,  97, 370, 566, 538,
       497, 284, 685, 358, 426, 580, 469, 342, 653, 608, 301, 510, 128,
       635,  98, 259, 454, 124, 258, 385, 663, 552, 594, 245, 202, 664,
       313, 273, 228, 177, 442, 229, 255, 131, 369, 176, 102, 217, 341,
       525, 691, 150, 687, 425, 387, 682, 636, 652, 537, 661, 161, 123,
       415, 149, 609, 189, 302, 714, 227, 397, 175, 565, 712, 690, 246,
       688, 103, 330, 711, 553,  99, 665, 610, 581, 593, 509, 622, 498,
       481, 274, 470, 122, 710, 651, 713, 554, 526, 190, 218,  96, 132,
       148, 162, 681, 680, 611, 692, 201, 679, 689, 592, 121, 100, 715,
       443, 359, 147, 719, 453, 499, 312, 101, 174, 709, 621,  95, 637,
       219, 331, 471,  72, 716,  70, 340, 120, 527, 508, 134, 303, 133,
       480,  71, 555, 650, 583, 717, 708, 582, 718, 163, 707, 247, 275,
       311,  69, 536, 639, 310, 396, 500, 283, 135, 220,  73, 146, 199,
       248, 678, 693,  74, 424, 221, 649, 638,  94, 564, 620, 191, 666,
       198, 368, 720, 226, 395, 173, 472, 339, 452, 722, 612, 584, 276,
       145, 556, 706, 105, 394, 282, 648, 338, 528, 423, 422, 444, 136,
       416, 705, 249, 694,  93, 164, 106, 254, 677, 192, 304, 104, 119,
       388, 360, 193, 723, 640,  68, 367, 451, 107, 332, 721, 200, 108,
       585, 613, 695,  92, 169, 168, 167, 170, 166, 172, 171, 165, 783,
       144,  37,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  38,
        25,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  26,  24,
       143,  11,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  12,
        23,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  49,  50,
        51, 112,  85,  86,  87,  88,  89,  90,  91, 109, 110, 111, 113,
        52, 114, 115, 116, 117, 137, 138, 139, 140, 141, 142,  84,  83,
        82,  81,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,
        64,  65,  66,  67,  75,  76,  77,  78,  79,  80, 118, 391, 194,
       728, 701, 702, 703, 704, 724, 725, 726, 727, 729, 699, 730, 731,
       732, 733, 734, 735, 736, 737, 700, 698, 739, 668, 641, 642, 643,
       644, 645, 646, 647, 667, 669, 697, 670, 671, 672, 673, 674, 675,
       676, 696, 738, 740, 195, 772, 764, 765, 766, 767, 768, 769, 770,
       771, 773, 762, 774, 775, 776, 777, 778, 779, 780, 781, 763, 761,
       741, 750, 742, 743, 744, 745, 746, 747, 748, 749, 751, 760, 752,
       753, 754, 755, 756, 757, 758, 759, 619, 618, 617, 366, 335, 336,
       337, 361, 362, 363, 364, 365, 389, 333, 390, 782, 392, 393, 417,
       418, 419, 420, 334, 309, 616, 252, 196, 197, 222, 223, 224, 225,
       250, 251, 253, 308, 277, 278, 279, 280, 281, 305, 306, 307, 421,
       445, 446, 562, 533, 534, 535, 557, 558, 559, 560, 561, 563, 447,
       586, 587, 588, 589, 590, 591, 614, 615, 532, 531, 530, 529, 448,
       449, 450, 473, 474, 475, 476, 477, 478, 479, 501, 502, 503, 504,
       505, 506, 507,   0])

S_hat_bart_mnist = ([610, 402, 292, 433, 403, 458, 456, 407, 320, 322, 464, 398, 375, 343, 293, 454, 405, 401, 376, 264, 239, 622, 542, 515, 513, 484, 457, 426, 369, 351, 348, 344, 321, 319, 596, 538, 489, 483, 460, 459, 429, 404, 400, 383, 374, 373, 347, 346, 317, 296, 234, 572, 525, 519, 455, 435, 379, 359, 342, 290, 265, 178, 597, 575, 556, 514, 512, 488, 486, 434, 432, 430, 412, 409, 389, 380, 361, 326, 325, 324, 302, 271, 235, 627, 602, 594, 555, 551, 529, 520, 518, 516, 490, 443, 438, 428, 411, 350, 349, 345, 331, 314, 301, 297, 274, 268, 263, 242, 216, 210, 651, 603, 580, 564, 544, 521, 511, 510, 492, 485, 461, 440, 427, 406, 399, 396, 377, 354, 353, 340, 329, 328, 318, 300, 266, 236, 159, 150, 728, 656, 652, 621, 599, 595, 574, 549, 548, 543, 540, 509, 495, 493, 487, 479, 469, 468, 467, 465, 462, 453, 439, 424, 417, 410, 387, 381, 356, 327, 305, 291, 256, 243, 218, 209, 189, 164, 155, 747, 679, 662, 632, 629, 625, 617, 598, 573, 571, 570, 557, 545, 536, 494, 491, 463, 451, 437, 431, 425, 415, 408, 397, 378, 371, 367, 357, 352, 341, 315, 298, 289, 286, 284, 273, 272, 267, 262, 261, 247, 245, 241, 240, 237, 227, 215, 214, 212, 207, 206, 188, 179, 160, 132, 128, 775, 705, 661, 654, 611, 585, 578, 568, 566, 550, 547, 535, 534, 524, 507, 500, 482, 481, 473, 452, 445, 444, 414, 392, 384, 382, 372, 368, 330, 313, 304, 299, 287, 285, 270, 269, 248, 233, 220, 217, 213, 205, 204, 192, 190, 187, 185, 183, 157, 139, 131, 63, 57, 29, 774, 767, 746, 743, 735, 721, 719, 718, 711, 703, 692, 687, 683, 682, 678, 668, 658, 657, 655, 650, 649, 647, 637, 635, 634, 633, 628, 626, 624, 623, 613, 590, 587, 583, 567, 559, 558, 541, 523, 517, 499, 480, 475, 471, 470, 449, 436, 422, 416, 394, 370, 366, 355, 338, 334, 333, 323, 316, 311, 295, 294, 288, 277, 258, 249, 246, 238, 231, 230, 229, 211, 181, 180, 168, 158, 154, 149, 130, 126, 110, 103, 102, 82, 72, 69, 18, 1, 780, 779, 777, 776, 773, 772, 758, 755, 750, 739, 738, 733, 725, 708, 704, 685, 684, 681, 675, 674, 673, 672, 653, 648, 645, 640, 639, 636, 631, 630, 619, 615, 612, 608, 607, 604, 600, 582, 579, 569, 554, 553, 552, 537, 533, 531, 526, 508, 506, 503, 466, 450, 442, 441, 423, 420, 413, 395, 385, 360, 339, 332, 310, 308, 303, 260, 259, 257, 254, 250, 244, 226, 222, 201, 199, 198, 191, 184, 169, 165, 163, 162, 156, 153, 143, 138, 137, 133, 127, 125, 124, 121, 115, 105, 101, 97, 95, 79, 78, 74, 73, 70, 67, 65, 62, 61, 52, 50, 49, 44, 43, 35, 34, 30, 19, 15, 8, 3, 783, 782, 781, 778, 766, 762, 760, 751, 749, 742, 740, 737, 731, 730, 729, 727, 713, 712, 709, 707, 698, 690, 688, 686, 680, 676, 671, 669, 666, 665, 664, 663, 638, 620, 618, 616, 609, 606, 605, 601, 593, 592, 591, 589, 584, 581, 576, 565, 563, 561, 560, 546, 539, 532, 528, 527, 522, 505, 498, 497, 496, 477, 476, 474, 472, 446, 419, 386, 364, 363, 362, 358, 335, 312, 283, 279, 278, 276, 275, 255, 251, 228, 225, 223, 219, 203, 200, 196, 195, 193, 186, 182, 175, 174, 173, 167, 161, 152, 151, 148, 146, 142, 141, 135, 120, 117, 113, 112, 87, 85, 81, 75, 71, 64, 58, 55, 54, 53, 46, 42, 40, 39, 38, 32, 31, 25, 24, 17, 16, 12, 10, 9, 6, 4, 2, 771, 770, 768, 764, 763, 761, 756, 754, 753, 752, 736, 734, 732, 726, 723, 722, 720, 717, 714, 710, 706, 702, 700, 699, 697, 696, 695, 694, 693, 691, 689, 677, 660, 659, 614, 588, 586, 562, 530, 504, 502, 501, 448, 447, 421, 418, 393, 391, 390, 388, 336, 309, 307, 306, 282, 252, 232, 221, 208, 197, 177, 176, 172, 171, 147, 140, 136, 134, 129, 123, 119, 114, 111, 108, 104, 98, 94, 90, 89, 88, 86, 84, 80, 77, 76, 68, 60, 51, 41, 37, 33, 28, 27, 22, 21, 20, 11, 0, 769, 765, 759, 757, 748, 745, 744, 741, 724, 716, 715, 701, 670, 667, 646, 644, 643, 642, 641, 577, 478, 365, 337, 281, 280, 253, 224, 202, 194, 170, 166, 145, 144, 122, 118, 116, 109, 107, 106, 100, 99, 96, 93, 92, 91, 83, 66, 59, 56, 48, 47, 45, 36, 26, 23, 14, 13, 7, 5])
# Just to compare what global features SHAP with DeepLift choose
X_train_ori =  loadmat("./mat_files/MNIST.mat")["train_x"].astype(np.float32)
x = np.array([X_train_ori[:,4],
              X_train_ori[:,15],
              X_train_ori[:,3]])
show_image(x, S_hat[0:300], "test_deeplift_300")

show_image(x, S_hat_bart_mnist[0:300], "test_bart_300")

with open('./parameters/MNIST_sorted.pkl', 'rb') as f:
    indices = pickle.load(f)

#show_image_old(x, indices['sorted_ind'], "test_nmp")
rows = [1, 3, 3, 2, 4, 2, 3, 4, 3, 5, 5, 3, 3, 5, 2, 4, 4, 4, 2, 6, 2, 4, 0, 1, 2, 3, 5, 1, 0, 0, 5, 0, 6, 2, 0, 6, 1, 5, 4, 5, 1, 0, 1, 6, 1, 6, 0, 6, 6]
cols = [4, 3, 1, 4, 2, 2, 4, 3, 5, 2, 4, 6, 2, 1, 3, 5, 4, 6, 1, 2, 6, 1, 1, 3, 5, 0, 3, 1, 6, 4, 6, 5, 4, 0, 2, 0, 5, 5, 0, 0, 0, 0, 2, 6, 6, 5, 3, 1, 3]

S_hat = []
patch_rows = 4
patch_cols = 4
for i in range(7*7):
    for j in range(patch_rows):
        for k in range(patch_cols):
            S_hat.append((patch_rows*rows[i] + k-1)*28 + patch_cols*cols[i] + j%4) 

show_image(x, S_hat[0:300], "test_nmp_300_4x4")


rows = [1,2,3,4,7,8,8,8,8,8,8,9,9,9,9,6,5,5,7,3,7,5,11,4
,7,6,12,5,11,9,3,6,6,10,6,4,7,10,7,6,3,6,1,2,4,12,5,10
,7,10,1,12,7,8,12,6,2,8,13,9,5,11,4,9,10,2,11,11,10,5,11,5
,4,12,7,13,1,3,9,4,6,4,5,1,9,12,12,2,12,0,0,4,1,4,8,5
,0,10,10,8,11,7,11,7,0,5,12,3,9,12,2,12,4,5,13,1,5,6,0,13
,10,2,4,5,9,0,7,8,0,1,7,3,10,9,11,4,4,11,8,10,3,6,2,7
,13,0,2,6,0,13,2,9,11,1,13,3,2,13,1,9,10,1,12,3,13,13,0,3
,10,11,6,12,8,2,6,10,11,13,0,13,3,2,13,1,2,3,0,13,1,1,11,0
,0,8,3,12]

cols = [4,4,4,4,4,4,8,9,10,11,12,5,6,7,8,7,9,6,2,8,6,11,4,7
,8,3,3,2,6,12,10,5,12,10,11,8,12,2,3,10,7,8,8,1,12,5,8,3
,5,8,5,4,7,6,13,4,11,3,13,10,5,8,5,4,5,12,1,3,11,3,5,1
,9,12,11,9,11,9,13,6,1,3,13,10,3,8,0,13,1,2,0,11,2,2,0,0
,12,6,13,13,7,1,0,9,10,10,9,1,1,7,2,6,1,12,1,1,4,0,13,11
,4,7,0,7,2,3,13,7,9,9,10,12,12,11,9,10,13,10,2,0,6,13,8,0
,12,8,6,6,5,0,5,9,11,6,2,11,3,8,13,0,1,0,11,13,5,3,1,3
,9,12,9,2,1,10,2,7,2,4,11,7,2,9,6,7,0,0,7,10,12,3,13,6
,4,5,5,10]

S_hat = []
patch_rows = 2
patch_cols = 2
for i in range(14*14):
    for j in range(patch_rows):
        for k in range(patch_cols):
            S_hat.append((patch_rows*rows[i] + k-1)*28 + patch_cols*cols[i] + j%4) 

show_image(x, S_hat[0:300], "test_nmp_300_2x2")

S_hat_lasso_mnist = np.array([430, 383, 402, 369, 214, 239, 158, 657, 185, 274, 442, 156, 548,
       184, 404, 608, 267, 607, 301, 425, 213, 211, 183, 512, 604, 566,
       539, 485, 273, 429, 663, 382, 218, 431, 272, 484, 549, 190, 240,
       441, 453, 630, 161, 397, 575, 520, 568, 540, 633, 513, 188, 187,
       634, 470, 246, 245, 496, 602, 576, 523, 538, 481, 266, 541, 302,
       215, 157, 414, 605, 408, 521, 511, 509, 552, 524, 328, 579, 580,
       294, 510, 412, 413, 405, 329, 212, 202, 457, 244, 159, 570, 571,
       572, 534, 533, 577, 573, 574, 578, 531, 530, 581, 532, 536, 554,
       567, 553, 555, 582, 551, 550, 558, 559, 569, 560, 547, 546, 543,
       542, 562, 564, 565, 561, 537,   0, 527, 434, 435, 436, 438, 439,
       440, 443, 444, 446, 447, 448, 449, 450, 452, 454, 455, 456, 433,
       458, 432, 426, 396, 398, 399, 400, 401, 403, 406, 407, 409, 410,
       411, 415, 419, 420, 421, 422, 424, 428, 528, 459, 466, 498, 499,
       500, 502, 503, 504, 505, 506, 508, 583, 514, 515, 518, 519, 522,
       525, 526, 497, 465, 495, 493, 467, 468, 469, 471, 472, 474, 475,
       476, 477, 478, 480, 482, 483, 486, 487, 491, 492, 494, 585, 596,
       587, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727,
       728, 729, 730, 731, 732, 733, 734, 715, 735, 714, 712, 693, 694,
       695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707,
       708, 709, 710, 711, 713, 736, 737, 738, 763, 764, 765, 766, 767,
       768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780,
       781, 762, 761, 760, 759, 739, 740, 741, 742, 743, 744, 745, 746,
       747, 692, 748, 750, 751, 752, 753, 754, 755, 756, 757, 758, 749,
       691, 690, 689, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626,
       627, 628, 629, 631, 632, 635, 636, 637, 638, 616, 615, 614, 613,
       588, 589, 590, 591, 592, 593, 594, 595, 395, 639, 597, 599, 600,
       601, 603, 606, 609, 610, 611, 612, 598, 586, 640, 642, 669, 670,
       671, 672, 673, 674, 675, 676, 677, 678, 679, 681, 682, 683, 684,
       685, 686, 687, 688, 668, 667, 666, 665, 643, 644, 645, 646, 647,
       648, 649, 650, 651, 641, 652, 654, 655, 656, 658, 659, 660, 661,
       662, 664, 653, 394, 391, 392, 122, 123, 124, 125, 126, 127, 128,
       129, 130, 121, 131, 133, 134, 135, 136, 137, 138, 139, 140, 141,
       132, 142, 120, 118,  98,  99, 100, 101, 102, 103, 104, 105, 106,
       119, 107, 109, 110, 111, 112, 113, 114, 115, 116, 117, 108,  97,
       143, 145, 176, 177, 178, 180, 181, 182, 186, 189, 191, 175, 192,
       194, 195, 196, 197, 198, 199, 200, 201, 203, 193, 144, 174, 172,
       146, 147, 148, 149, 150, 151, 153, 154, 155, 173, 160, 163, 164,
       165, 166, 167, 168, 169, 170, 171, 162,  96,  95,  94,  25,  26,
        27,  28,  29,  30,  31,  32,  33,  24,  34,  36,  37,  38,  39,
        40,  41,  42,  43,  44,  35,  45,  23,  21,   1,   2,   3,   4,
         5,   6,   7,   8,   9,  22,  10,  12,  13,  14,  15,  16,  17,
        18,  19,  20,  11,  46,  47,  48,  74,  75,  76,  77,  78,  79,
        80,  81,  82,  73,  83,  85,  86,  87,  88,  89,  90,  91,  92,
        93,  84,  72,  71,  70,  49,  50,  51,  52,  53,  54,  55,  56,
        57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
       393, 205, 204, 308, 340, 339, 338, 337, 336, 335, 334, 331, 330,
       327, 322, 321, 315, 314, 313, 312, 311, 286, 287, 292, 293, 295,
       296, 341, 297, 300, 303, 306, 307, 309, 310, 299, 342, 343, 346,
       373, 374, 375, 376, 377, 378, 372, 379, 381, 384, 385, 386, 387,
       782, 380, 285, 371, 368, 352, 353, 354, 355, 356, 357, 370, 358,
       362, 363, 364, 365, 366, 367, 359, 284, 783, 257, 262, 260, 259,
       223, 258, 236, 224, 225, 226, 256, 255, 227, 228, 229, 242, 230,
       254, 283, 253, 252, 251, 250, 231, 248, 247, 232, 243, 221, 220,
       222, 217, 271, 276, 219, 238, 241, 270, 278, 209, 269, 210, 275,
       208, 237, 268, 279, 280, 265, 281, 264, 282, 216, 320, 288, 323,
       416, 556, 332, 423, 206, 316, 479, 437, 344, 360, 291, 326, 351,
       427, 451, 544, 345, 516, 388, 304, 289, 235, 233, 390, 517, 261,
       234, 545, 290, 490, 298, 460, 319, 207, 152, 535, 318, 489, 488,
       324, 462, 418, 317, 263, 179, 417, 350, 461, 464, 563, 277, 389,
       325, 333, 507, 584, 305, 463, 347, 361, 348, 349, 445, 473, 529,
       680, 557, 249, 501])

show_image(x, S_hat_lasso_mnist[0:300], "test_lasso_300")


S_hat_rf_mnist = np.array([102, 123, 130, 131, 132, 133, 148, 149, 150, 152, 153, 154, 155,
       156, 157, 158, 159, 160, 161, 162, 163, 164, 174, 175, 176, 177,
       178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190,
       191, 192, 193, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210,
       211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 229, 230,
       231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243,
       244, 245, 246, 247, 248, 249, 255, 256, 257, 258, 259, 260, 261,
       262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274,
       275, 276, 277, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292,
       293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305,
       311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323,
       324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 339, 340, 341,
       342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354,
       355, 356, 357, 358, 359, 360, 361, 367, 368, 369, 370, 371, 372,
       373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385,
       386, 387, 388, 389, 395, 396, 397, 398, 399, 400, 401, 402, 403,
       404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416,
       417, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434,
       435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 451, 452,
       453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465,
       466, 467, 468, 469, 470, 471, 472, 473, 479, 480, 481, 482, 483,
       484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496,
       497, 498, 499, 500, 501, 507, 508, 509, 510, 511, 512, 513, 514,
       515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527,
       528, 529, 530, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544,
       545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 563,
       564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576,
       577, 578, 579, 580, 581, 582, 583, 584, 586, 591, 592, 593, 594,
       595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607,
       608, 609, 610, 611, 620, 621, 622, 623, 624, 625, 626, 627, 628,
       629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 649, 650, 651,
       652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 665,
       678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690,
       691, 707])

show_image(x, S_hat_rf_mnist[0:300], "test_rf_300")