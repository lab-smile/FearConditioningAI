M = readmatrix('vgg/Conditioning_Orientation_3D_VGG_BN.xlsx');

contrast = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
orientation = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180];

%surfl(contrast, orientation, M(:, :))
%colormap(pink)
%shading interp

[x,y] = meshgrid(contrast, orientation);
BDmatrixq = interp2(contrast, orientation, M, x, y, 'cubic');
[c,h] = contourf(x,y,BDmatrixq,50);
set(h,'edgecolor', 'none')
colormap('parula');
colorbar
xlabel('Contrast (%)')
ylim([0 180])
ylabel('Orientation (degree)')
zlabel('Valence Score')
grid on

%%
%M = readmatrix('Experiment9.xlsx', 'Sheet','freq_ori');
M = readmatrix('vgg/Conditioning_Frequency_3D_VGG_BN.xlsx');

freq = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16 0.18, 0.2, 0.22, 0.24, 0.26,0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.4];
orientation = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180];

surfl(orientation, freq, M(:, :))
colormap(pink)
shading interp
xlabel('Orientation (degree)')
ylim()
ylabel('Frequency')
zlabel('Valence Score')
grid on

%%

%M = readmatrix('Experiment9.xlsx', 'Sheet','freq_cont');
M = readmatrix('Conditioning_Frequency_3D.xlsx');
freq = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16 0.18, 0.2, 0.22, 0.24, 0.26,0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.4];
contrast = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

surfl(contrast, freq, M(:, :)')
colormap(pink)
shading interp
xlabel('Contrast(%)')
ylim()
ylabel('Frequency')
zlabel('Valence Score')
grid on

%% Experiment 10

M = readmatrix('Conditioning_Contrast_3D.xlsx');

freq = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16 0.18, 0.2, 0.22, 0.24, 0.26,0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.4];
contrast = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

surfl(contrast, freq, M(:, :)')
colormap(pink)
shading interp
xlabel('Contrast(%)')
ylim()
ylabel('Frequency')
zlabel('Valence Score')
grid on

%% Orientation topography

M = readmatrix('/vgg/Conditioning_Orientation_3D_VGG_FC2.xlsx');
M=M';
contrast = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
orientation = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180];

%surfl(contrast, orientation, M(:, :))
%colormap(pink)
%shading interp

[x,y] = meshgrid(contrast, orientation);
BDmatrixq = interp2(contrast, orientation, M', x, y, 'cubic');
[c,h] = contourf(x,y,BDmatrixq,100);
set(h,'edgecolor', 'none')
colormap('parula');
colorbar
caxis([1 9])
xlabel('Contrast (%)','FontSize',18)
ylim([0 180])
ylabel('Orientation (degree)','FontSize',18)
zlabel('Valence Score')
grid on

ax=gca;
ax.FontSize = 18;

x0=10;
y0=10;
width=900;
height=400;
set(gcf,'position',[x0,y0,width,height])

%% Frequency topography

M = readmatrix('/vgg/Conditioning_Frequency_3D_VGG_FC2.xlsx');

freq = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16 0.18, 0.2, 0.22, 0.24, 0.26,0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.4];
contrast = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

%surfl(contrast, orientation, M(:, :))
%colormap(pink)
%shading interp

[x,y] = meshgrid(contrast, freq);
BDmatrixq = interp2(contrast, freq, M', x, y, 'cubic');
[c,h] = contourf(x,y,BDmatrixq,50);
set(h,'edgecolor', 'none')
colormap('parula');
colorbar
caxis([1 9])
xlabel('Contrast(%)','FontSize',18)
ylim()
ylabel('Frequency','FontSize',18)
zlabel('Valence Score')
grid on

ax=gca;
ax.FontSize = 18;

x0=10;
y0=10;
width=900;
height=400;
set(gcf,'position',[x0,y0,width,height])

%% Contrast topography

M = readmatrix('/vgg/Conditioning_Contrast_3D_VGG_FC2.xlsx');

freq = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16 0.18, 0.2, 0.22, 0.24, 0.26,0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.4];
contrast = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

%surfl(contrast, orientation, M(:, :))
%colormap(pink)
%shading interp

[x,y] = meshgrid(freq, contrast);
BDmatrixq = interp2(freq, contrast, M, x, y, 'cubic');
[c,h] = contourf(x,y,BDmatrixq,50);
set(h,'edgecolor', 'none')
colormap('parula');
colorbar
caxis([1 9])
ylabel('Contrast(%)','FontSize',18)
xlim()
xlabel('Frequency','FontSize',18)
zlabel('Valence Score')
grid on

ax=gca;
ax.FontSize = 18;

x0=10;
y0=10;
width=900;
height=400;
set(gcf,'position',[x0,y0,width,height])


