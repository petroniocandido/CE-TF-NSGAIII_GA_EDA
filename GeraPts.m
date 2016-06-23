
% Script para gera��o de pontos de referencia estruturados
% --------------------------------------------------------

% MODIFICAR o n�mero de "m" e n�mero de divis�es "p" desejados

% ### ATENCAO - A GERA��O DE PONTOS PARA UM ESPACO DE GRANDE NUMERO DE OBJETIVOS
% ### E OU CONSIDERANDO MUITAS DIVISOES POR OBJETIVO PODE DEMANDAR ALTO TEMPO 
% ### DE PROCESSAMENTO --- CASO DESEJADO INTERROMPER O PROCESSO DE GERA��O PRESSIONE "CTRL+C"

% numero de objetivos
m=3; 

% numero de divisoes
p=12;

% tempo inicial
tic

% vetor base - pontos em cada objetivo
w_base = linspace(0,1,p+1);

% conjunto de espa�os dimensionais discretizados 
wi = repmat(w_base,1,m);

% realiza combina��o entre todas as dimens�es, para cada objetivo 'm'
W = combnk(wi,m);

% elimina linhas duplicadas
W = unique(W,'rows');

% mant�m apenas vetores no hiperplano de valor unit�rio
W = W(abs(sum(W,2)-1)<=eps(1),:)

% n�mero de pontos gerados
H = length(W)

% plota pontos do espaco objetivo
if m==3
    figure()
    plot3(W(:,1),W(:,2),W(:,3),'or');
end

toc