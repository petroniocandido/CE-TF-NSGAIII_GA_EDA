
% Fun��o para leitura dos pontos de refer�ncia
function [Zs,nZs] = ptos_referencia(m)

% gera pontos de referencia conforme aqueles pontos usados nos problemas DTLZ1 e
% DTLZ2 pelos artigos originais do NSGAIII e MOEA-DD

% entradas:
% m    -> n�mero de objetivos

% sa�das:
% Zs   -> pontos de refer�ncia
% H    -> numero de pontos de refer�ncia

% DTLZ1 & DTLZ2 -> 3 objetivos e 12 divis�es
if m==3

    % numero de pontos
    nZs=91;

    % pontos de referencia
    Zs = [...
      0    0      1;...
      0    1/12   11/12;...
      0    1/6    5/6;...
      0    1/4    3/4;...
      0    1/3    2/3;...
      0    5/12   7/12;...
      0    1/2    1/2;...
      0    7/12   5/12;...
      0    2/3    1/3;...
      0    3/4    1/4;...
      0    5/6    1/6;...
      0    11/12  1/12;...
      0    1      0;...
      1/12    0      11/12;...
      1/12    1/12   5/6;...
      1/12    1/6    3/4;...
      1/12    1/4    2/3;...
      1/12    1/3    7/12;...
      1/12    5/12   1/2;...
      1/12    1/2    5/12;...
      1/12    7/12   1/3;...
      1/12    2/3    1/4;...
      1/12    3/4    1/6;...
      1/12    5/6    1/12;...
      1/12   11/12     0;...
      1/6    0       5/6;...
      1/6    1/12    3/4;...
      1/6    1/6    2/3;...
      1/6    1/4    7/12;...
      1/6    1/3    1/2;...
      1/6    5/12    5/12;...
      1/6    1/2    1/3;...
      1/6    7/12    1/4;...
      1/6    2/3    1/6;...
      1/6    3/4    1/12;...
      1/6    5/6         0;...
      1/4    0    3/4;...
      1/4    1/12    2/3;...
      1/4    1/6    7/12;...
      1/4    1/4    1/2;...
      1/4    1/3    5/12;...
      1/4    5/12    1/3;...
      1/4    1/2    1/4;...
      1/4    7/12    1/6;...
      1/4    2/3    1/12;...
      1/4    3/4         0;...
      1/3    0    2/3;...
      1/3    1/12    7/12;...
      1/3    1/6    1/2;...
      1/3    1/4    5/12;...
      1/3    1/3    1/3;...
      1/3    5/12    1/4;...
      1/3    1/2    1/6;...
      1/3    7/12    1/12;...
      1/3    2/3         0;...
      5/12   0    7/12;...
      5/12   1/12    1/2;...
      5/12   1/6    5/12;...
      5/12   1/4    1/3;...
      5/12   1/3    1/4;...
      5/12   5/12    1/6;...
      5/12   1/2    1/12;...
      5/12   7/12         0;...
      1/2    0    1/2;...
      1/2    1/12    5/12;...
      1/2    1/6    1/3;...
      1/2    1/4    1/4;...
      1/2    1/3    1/6;...
      1/2    5/12    1/12;...
      1/2    1/2         0;...
      7/12   0    5/12;...
      7/12   1/12    1/3;...
      7/12   1/6    1/4;...
      7/12   1/4    1/6;...
      7/12   1/3    1/12;...
      7/12   5/12         0;...
      2/3    0    1/3;...
      2/3    1/12    1/4;...
      2/3    1/6    1/6;...
      2/3    1/4    1/12;...
      2/3    1/3         0;...
      3/4    0    1/4;...
      3/4    1/12    1/6;...
      3/4    1/6    1/12;...
      3/4    1/4         0;...
      5/6    0    1/6;...
      5/6    1/12    1/12;...
      5/6    1/6         0;...
      11/12  0    1/12;...
      11/12  1/12         0;...
      1      0         0 ...
    ];    

% DTLZ1 & DTLZ2 -> 5 objetivos e 6 divis�es
elseif m==5

        % n�mero de pontos
        nZs=210;

        % pontos de referencia
        Zs = [...
         0         0         0         0    1;...
         0         0         0    1/6    5/6;...
         0         0         0    1/3    2/3;...
         0         0         0    1/2    1/2;...
         0         0         0    2/3    1/3;...
         0         0         0    5/6    1/6;...
         0         0         0    1         0;...
         0         0    1/6         0    5/6;...
         0         0    1/6    1/6    2/3;...
         0         0    1/6    1/3    1/2
         0         0    1/6    1/2    1/3;...
         0         0    1/6    2/3    1/6;...
         0         0    1/6    5/6         0;...
         0         0    1/3         0    2/3;...
         0         0    1/3    1/6    1/2;...
         0         0    1/3    1/3    1/3;...
         0         0    1/3    1/2    1/6;...
         0         0    1/3    2/3         0;...
         0         0    1/2         0    1/2;...
         0         0    1/2    1/6    1/3;...
         0         0    1/2    1/3    1/6;...
         0         0    1/2    1/2         0;...
         0         0    2/3         0    1/3;...
         0         0    2/3    1/6    1/6;...
         0         0    2/3    1/3         0;...
         0         0    5/6         0    1/6;...
         0         0    5/6    1/6         0;...
         0         0    1         0         0;...
         0    1/6         0         0    5/6;...
         0    1/6         0    1/6    2/3;...
         0    1/6         0    1/3    1/2;...
         0    1/6         0    1/2    1/3;...
         0    1/6         0    2/3    1/6;...
         0    1/6         0    5/6         0;...
         0    1/6    1/6         0    2/3;...
         0    1/6    1/6    1/6    1/2;...
         0    1/6    1/6    1/3    1/3;...
         0    1/6    1/6    1/2    1/6;...
         0    1/6    1/6    2/3         0;...
         0    1/6    1/3         0    1/2;...
         0    1/6    1/3    1/6    1/3;...
         0    1/6    1/3    1/3    1/6;...
         0    1/6    1/3    1/2         0;...
         0    1/6    1/2         0    1/3;...
         0    1/6    1/2    1/6    1/6;...
         0    1/6    1/2    1/3         0;...
         0    1/6    2/3         0    1/6;...
         0    1/6    2/3    1/6         0;...
         0    1/6    5/6         0         0;...
         0    1/3         0         0    2/3;...
         0    1/3         0    1/6    1/2;...
         0    1/3         0    1/3    1/3;...
         0    1/3         0    1/2    1/6;...
         0    1/3         0    2/3         0;...
         0    1/3    1/6         0    1/2;...
         0    1/3    1/6    1/6    1/3;...
         0    1/3    1/6    1/3    1/6;...
         0    1/3    1/6    1/2         0;...
         0    1/3    1/3         0    1/3;...
         0    1/3    1/3    1/6    1/6;...
         0    1/3    1/3    1/3         0;...
         0    1/3    1/2         0    1/6;...
         0    1/3    1/2    1/6    0.0000;...
         0    1/3    2/3         0         0;...
         0    1/2         0         0    1/2;...
         0    1/2         0    1/6    1/3;...
         0    1/2         0    1/3    1/6;...
         0    1/2         0    1/2         0;...
         0    1/2    1/6         0    1/3;...
         0    1/2    1/6    1/6    1/6;...
         0    1/2    1/6    1/3         0;...
         0    1/2    1/3         0    1/6;...
         0    1/2    1/3    1/6    0.0000;...
         0    1/2    1/2         0         0;...
         0    2/3         0         0    1/3;...
         0    2/3         0    1/6    1/6;...
         0    2/3         0    1/3         0;...
         0    2/3    1/6         0    1/6;...
         0    2/3    1/6    1/6         0;...
         0    2/3    1/3         0         0;...
         0    5/6         0         0    1/6;...
         0    5/6         0    1/6         0;...
         0    5/6    1/6         0         0;...
         0    1         0         0         0;...
    1/6         0         0         0    5/6;...
    1/6         0         0    1/6    2/3;...
    1/6         0         0    1/3    1/2;...
    1/6         0         0    1/2    1/3;...
    1/6         0         0    2/3    1/6;...
    1/6         0         0    5/6         0;...
    1/6         0    1/6         0    2/3;...
    1/6         0    1/6    1/6    1/2;...
    1/6         0    1/6    1/3    1/3;...
    1/6         0    1/6    1/2    1/6;...
    1/6         0    1/6    2/3         0;...
    1/6         0    1/3         0    1/2;...
    1/6         0    1/3    1/6    1/3;...
    1/6         0    1/3    1/3    1/6;...
    1/6         0    1/3    1/2         0;...
    1/6         0    1/2         0    1/3;...
    1/6         0    1/2    1/6    1/6;...
    1/6         0    1/2    1/3         0;...
    1/6         0    2/3         0    1/6;...
    1/6         0    2/3    1/6         0;...
    1/6         0    5/6         0         0;...
    1/6    1/6         0         0    2/3;...
    1/6    1/6         0    1/6    1/2;...
    1/6    1/6         0    1/3    1/3;...
    1/6    1/6         0    1/2    1/6;...
    1/6    1/6         0    2/3         0;...
    1/6    1/6    1/6         0    1/2;...
    1/6    1/6    1/6    1/6    1/3;...
    1/6    1/6    1/6    1/3    1/6;...
    1/6    1/6    1/6    1/2         0;...
    1/6    1/6    1/3         0    1/3;...
    1/6    1/6    1/3    1/6    1/6;...
    1/6    1/6    1/3    1/3         0;...
    1/6    1/6    1/2         0    1/6;...
    1/6    1/6    1/2    1/6    0.0000;...
    1/6    1/6    2/3         0         0;...
    1/6    1/3         0         0    1/2;...
    1/6    1/3         0    1/6    1/3;...
    1/6    1/3         0    1/3    1/6;...
    1/6    1/3         0    1/2         0;...
    1/6    1/3    1/6         0    1/3;...
    1/6    1/3    1/6    1/6    1/6;...
    1/6    1/3    1/6    1/3         0;...
    1/6    1/3    1/3         0    1/6;...
    1/6    1/3    1/3    1/6    0.0000;...
    1/6    1/3    1/2         0         0;...
    1/6    1/2         0         0    1/3;...
    1/6    1/2         0    1/6    1/6;...
    1/6    1/2         0    1/3         0;...
    1/6    1/2    1/6         0    1/6;...
    1/6    1/2    1/6    1/6    0.0000;...
    1/6    1/2    1/3         0         0;...
    1/6    2/3         0         0    1/6;...
    1/6    2/3         0    1/6         0;...
    1/6    2/3    1/6         0         0;...
    1/6    5/6         0         0         0;...
    1/3         0         0         0    2/3;...
    1/3         0         0    1/6    1/2;...
    1/3         0         0    1/3    1/3;...
    1/3         0         0    1/2    1/6;...
    1/3         0         0    2/3         0;...
    1/3         0    1/6         0    1/2;...
    1/3         0    1/6    1/6    1/3;...
    1/3         0    1/6    1/3    1/6;...
    1/3         0    1/6    1/2         0;...
    1/3         0    1/3         0    1/3;...
    1/3         0    1/3    1/6    1/6;...
    1/3         0    1/3    1/3         0;...
    1/3         0    1/2         0    1/6;...
    1/3         0    1/2    1/6    0.0000;...
    1/3         0    2/3         0         0;...
    1/3    1/6         0         0    1/2;...
    1/3    1/6         0    1/6    1/3;...
    1/3    1/6         0    1/3    1/6;...
    1/3    1/6         0    1/2         0;...
    1/3    1/6    1/6         0    1/3;...
    1/3    1/6    1/6    1/6    1/6;...
    1/3    1/6    1/6    1/3         0;...
    1/3    1/6    1/3         0    1/6;...
    1/3    1/6    1/3    1/6    0.0000;...
    1/3    1/6    1/2         0         0;...
    1/3    1/3         0         0    1/3;...
    1/3    1/3         0    1/6    1/6;...
    1/3    1/3         0    1/3         0;...
    1/3    1/3    1/6         0    1/6;...
    1/3    1/3    1/6    1/6    0.0000;...
    1/3    1/3    1/3         0         0;...
    1/3    1/2         0         0    1/6;...
    1/3    1/2         0    1/6    0.0000;...
    1/3    1/2    1/6         0    0.0000;...
    1/3    2/3         0         0         0;...
    1/2         0         0         0    1/2;...
    1/2         0         0    1/6    1/3;...
    1/2         0         0    1/3    1/6;...
    1/2         0         0    1/2         0;...
    1/2         0    1/6         0    1/3;...
    1/2         0    1/6    1/6    1/6;...
    1/2         0    1/6    1/3         0;...
    1/2         0    1/3         0    1/6;...
    1/2         0    1/3    1/6    0.0000;...
    1/2         0    1/2         0         0;...
    1/2    1/6         0         0    1/3;...
    1/2    1/6         0    1/6    1/6;...
    1/2    1/6         0    1/3         0;...
    1/2    1/6    1/6         0    1/6;...
    1/2    1/6    1/6    1/6    0.0000;...
    1/2    1/6    1/3         0         0;...
    1/2    1/3         0         0    1/6;...
    1/2    1/3         0    1/6    0.0000;...
    1/2    1/3    1/6         0    0.0000;...
    1/2    1/2         0         0         0;...
    2/3         0         0         0    1/3;...
    2/3         0         0    1/6    1/6;...
    2/3         0         0    1/3         0;...
    2/3         0    1/6         0    1/6;...
    2/3         0    1/6    1/6         0;...
    2/3         0    1/3         0         0;...
    2/3    1/6         0         0    1/6;...
    2/3    1/6         0    1/6         0;...
    2/3    1/6    1/6         0         0;...
    2/3    1/3         0         0         0;...
    5/6         0         0         0    1/6;...
    5/6         0         0    1/6         0;...
    5/6         0    1/6         0         0;...
    5/6    1/6         0         0         0;...
    1         0         0         0         0;...
    ];
      
else
    error('PONTOS DE REFERENCIA N�O EXISTEM PARA O NUMERO DE OBJETIVOS E NUMERO DE DIVISOES REQUERIDOS!!!!')    
end

end