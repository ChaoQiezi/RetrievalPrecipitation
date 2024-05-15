clc
clear
close all
file_path='C:\Users\chfs1\Desktop\闲鱼\20';
cd(file_path)
file1='2010-2012.nc';
file2='2013-2015.nc';
ncdisp(file1)
info=ncinfo(file1);

%
lon=ncread(file1,'longitude');
lat=ncread(file1,'latitude');
time1=ncread(file1,'time');
time2=ncread(file2,'time');
time=[time1;time2];
t2m1=ncread(file1,'t2m');
t2m2=ncread(file2,'t2m');
t2m=cat(3,t2m1,t2m2);
skt1=ncread(file1,'skt');
skt2=ncread(file2,'skt');
skt=cat(3,skt1,skt2);

% 转换时间格式
t0=datetime(1900,1,1,0,0,0);
t=t0+double(time(:))/24;
% t=datestr(t,31);%将datetime时间转换为字符串时间以便输出到excel

%生成网格
[lat2,lon2]=meshgrid(lat,lon);
lat2=double(lat2);
lon2=double(lon2);

%%
% 计算每天的平均值
t2m_daily_mean = zeros(5, 2, size(t,1)/24);  % 创建一个大小为 10x5x365 的数组来存储结果
skt_daily_mean = zeros(5, 2, size(t,1)/24);
day_count = 1;  % 记录当前的天数

for i = 1:24:size(t,1)
    % 从原始数组中提取当前一天的数据
    day_data_t2m = t2m(:, :, i:i+23);
    day_data_skt = skt(:, :, i:i+23);

    % 计算当前一天的平均值，并存储到结果数组中
    t2m_daily_mean(:, :, day_count) = mean(day_data_t2m, 3);
    skt_daily_mean(:, :, day_count) = mean(day_data_skt, 3);

    % 增加天数计数器
    day_count = day_count + 1;
end

%% 插值

%  p1:工布江达（93.25,29.883）
%  p2:更张（94.15,29.75）
%  p3:巴河桥（93.6667,29.8667）

p_lon(1,1)=93.25;
p_lon(2,1)=94.15;
p_lon(3,1)=93.6667;

p_lat(1,1)=29.883;
p_lat(2,1)=29.75;
p_lat(3,1)=29.8667;

for p = 1:3
    for day =1:size(t,1)/24
        t2m_interp(day,p) = griddata(lon2,lat2,t2m_daily_mean(:,:,day),p_lon(p),p_lat(p));
        skt_interp(day,p) = griddata(lon2,lat2,skt_daily_mean(:,:,day),p_lon(p),p_lat(p));
    end
end

%% 输出

st=datetime(2010,1,1);
et=datetime(2015,12,31);
date_vec=datestr(datevec(st:et),29);

T=cell(size(t,1)/24,1);%将时间放入cell中，方便输出到excel
for i=1:size(t,1)/24
    T{i,1}=date_vec(i,:);
end

A{1}='时间';A{2}='2 metre temperature';A{3}='Skin temperature';

excel_filename='data.xlsx';
xlswrite(excel_filename, A ,'工布江达','A1');
xlswrite(excel_filename, T ,'工布江达','A2');
xlswrite(excel_filename, t2m_interp(:,1) ,'工布江达','B2');
xlswrite(excel_filename, skt_interp(:,1) ,'工布江达','C2');

xlswrite(excel_filename, A ,'更张','A1');
xlswrite(excel_filename, T ,'更张','A2');
xlswrite(excel_filename, t2m_interp(:,2) ,'更张','B2');
xlswrite(excel_filename, skt_interp(:,2) ,'更张','C2');

xlswrite(excel_filename, A ,'巴河桥','A1');
xlswrite(excel_filename, T ,'巴河桥','A2');
xlswrite(excel_filename, t2m_interp(:,3) ,'巴河桥','B2');
xlswrite(excel_filename, skt_interp(:,3) ,'巴河桥','C2');

