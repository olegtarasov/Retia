using MathNet.Numerics.LinearAlgebra.Single;
using Retia.Mathematics;

namespace Retia.Tests.Mathematics
{
    public partial class MatrixTests
    {
        protected static Matrix Matrix5By3 => MatrixFactory.ParseString(@"0,48867764207007 4,83962335616333 -3,74576540605434 
                                                        -1,11781515465948 -0,64865443187238 2,21479775254372 
                                                        -1,65766293958652 0,11036635148822 -1,93700658014836 
                                                        -4,11476715426648 1,71772796042158 0,32568804702055 
                                                        -1,37108411470944 -4,62275301088707 1,8326839277673 ");

        protected static Matrix Matrix3By6 => MatrixFactory.ParseString(@"-0,14720623621121 4,87011260812642 4,46601549138595 4,78338971258299 -4,46473619875719 0,10465222648562 
                                                            -1,63701515953849 3,10175835532218 -3,00050672050589 2,26877169556393 -2,77909147915388 -4,67759371254481 
                                                            2,17523330691049 0,78365747620522 -2,87786038959299 2,66903196818616 3,49864778038983 -1,37198884336836 ");

        protected static Matrix Matrix3By6_2 => MatrixFactory.ParseString(@"-4,28321383860112 -1,35080893354062 -3,22543903171338 3,95698704242566 2,66888237915415 1,83179720157375 
                                                            1,0609359834627 1,08751577142976 -3,95153975996726 -0,79624486891378 -1,4817940334239 0,72669421123653 
                                                            -2,7466795489875 -1,75442705245429 1,42283824105879 1,71571644335786 0,53173671734135 -2,41823336920619");

        protected static Matrix ColumnVector5 => MatrixFactory.ParseString(@"3,32129768483401 
                                                            0,64057354146641 
                                                            3,63054845884002 
                                                            4,96846716383866 
                                                            -0,77085907839744");

        protected static Matrix ColumnVector5_2 => MatrixFactory.ParseString(@"3,48713505011384 
                                                                -2,11142700962323 
                                                                2,73935758869134 
                                                                2,5506575021663 
                                                                0,77108238626788");

        protected static Matrix RowVector5 => MatrixFactory.ParseString(@"-0,71367357657928 3,35330983081521 -0,16776382698108 -3,87399159319419 -0,91424474768073");

        protected static Matrix RowVector5_2 => MatrixFactory.ParseString(@"1,81995936521327 1,47210847887774 0,16952450627905 1,69332691779981 -4,04017075385906");

        protected static Matrix ColumnVector3 => MatrixFactory.ParseString(@"-1,81991562564853 
                                                            4,20926216487273 
                                                            2,62784341705397");

        protected static Matrix RowVector3 => MatrixFactory.ParseString(@"4,71388211460499 0,18313295914938 3,04326736277121");

        protected static Matrix ColumnVector6 => MatrixFactory.ParseString(@"-0,68153440285546 
                                                            -2,47425174222991 
                                                            -2,17648161909379 
                                                            2,01797706401813 
                                                            0,2224841225997 
                                                            -1,54204056437222 
                                                            ");
    }
}