%simulation parameters
fs = 50e5;                 
Ts = 1/fs;
N =1e6;
time_domain=linspace(0,N/fs,N);
frequency_domain=linspace(-fs/2,fs/2,length(time_domain));
channel_band_width=1e5;
Tb=2/channel_band_width;
Rb=1/Tb;
%% 
%%%%%%%%%%%%%% part 1 %%%%%%%%%%%%%%%%%

band_limited_channel_in_frequency_domain=zeros(1,length(frequency_domain));
c=((length(frequency_domain))/fs)*(fs/2-channel_band_width);
d=((length(frequency_domain))/fs)*(fs/2+channel_band_width);
e=round(c);
f=round(d);
band_limited_channel_in_frequency_domain(e:f)=1;
one_square_pulse_in_time_domain=zeros(1,length(time_domain));
s=((Tb*length(time_domain)/(N/fs)));
one_square_pulse_in_time_domain(1:s)=1;
one_square_pulse_in_frequency_domain=(fftshift(fft(one_square_pulse_in_time_domain)));
received_one_square_pulse_in_frequency_domain=one_square_pulse_in_frequency_domain.*band_limited_channel_in_frequency_domain;
received_one_square_pulse_in_time_domain=real(ifft(ifftshift(received_one_square_pulse_in_frequency_domain)));
second_square_pulse=zeros(1,length(time_domain));
second_square_pulse(s:2*s)=1;
received_second_square_pulse_in_time_domain=real(ifft(ifftshift((fftshift(fft(second_square_pulse))).*band_limited_channel_in_frequency_domain)));
pulse_shaped_signal_in_frequency_domain=zeros(1,length(frequency_domain));
c1=((length(frequency_domain))/fs)*(fs/2-Rb/2);
d1=((length(frequency_domain))/fs)*(fs/2+Rb/2);
e1=round(c1);
f1=round(d1);
pulse_shaped_signal_in_frequency_domain(e1:f1)=Tb;
pulse_shaped_signal_in_time_domain=real(ifft(ifftshift(pulse_shaped_signal_in_frequency_domain)));
second_pulse_shaped_signal_in_frequency_domain=exp(-2*pi*1i*Tb.*frequency_domain).*pulse_shaped_signal_in_frequency_domain;
second_pulse_shaped_signal_in_time_domain=real(ifft(ifftshift(second_pulse_shaped_signal_in_frequency_domain)));
recevied_pulse_shaped_signal_in_frequency_domain=pulse_shaped_signal_in_frequency_domain.*band_limited_channel_in_frequency_domain;
recevied_pulse_shaped_signal_in_time_domain=real(ifft(ifftshift(recevied_pulse_shaped_signal_in_frequency_domain)));
recevied_second_pulse_shaped_signal_in_frequency_domain=second_pulse_shaped_signal_in_frequency_domain.*band_limited_channel_in_frequency_domain;
recevied_second_pulse_shaped_signal_in_time_domain=real(ifft(ifftshift(recevied_second_pulse_shaped_signal_in_frequency_domain)));



%%%%%%%%%%%%%%plotting%%%%%%%%%%%%%%

f1=figure;
figure(f1);
subplot(2,1,1)
suptitle('signal before channel');
plot(time_domain(1:4*s),one_square_pulse_in_time_domain(1:4*s),'linewidth',2);
title('One square pulse in time domain','linewidth',2);
xlabel('Time domain','linewidth',2);
ylabel('Amplitude','linewidth',2);
grid on
subplot(2,1,2)
plot(frequency_domain,real(one_square_pulse_in_frequency_domain),'linewidth',2);
xlim([-5 5]*1e5)
title('One square pulse in frequency domain','linewidth',2);
xlabel('Frequency domain','linewidth',2);
ylabel('Amplitude','linewidth',2);
grid on

f2=figure;
figure(f2);
subplot(2,1,1)
suptitle('signal after channel');
plot(time_domain(1:4*s),received_one_square_pulse_in_time_domain(1:4*s),'linewidth',2);
title('One square pulse in time domain','linewidth',2);
xlabel('Time domain','linewidth',2);
ylabel('Amplitude','linewidth',2);
grid on
subplot(2,1,2)
plot(frequency_domain,real(received_one_square_pulse_in_frequency_domain),'linewidth',2);
xlim([-5 5]*1e5)
title('One square pulse in frequency domain','linewidth',2);
xlabel('Frequency domain','linewidth',2);
ylabel('Amplitude','linewidth',2);
grid on

f3=figure;
figure(f3);
subplot(2,1,1)
suptitle('consecutive square pulse in time domain');
plot(time_domain(1:10*s),one_square_pulse_in_time_domain(1:10*s),'linewidth',2);
hold on
plot(time_domain(1:10*s),second_square_pulse(1:10*s),'r','linewidth',2);
hold on
title('before channel','linewidth',2);
xlabel('Time domain','linewidth',2);
ylabel('Amplitude','linewidth',2);
grid on
subplot(2,1,2)
plot(time_domain(1:10*s),received_one_square_pulse_in_time_domain(1:10*s),'linewidth',2);
hold on
plot(time_domain(1:10*s),received_second_square_pulse_in_time_domain(1:10*s),'r','linewidth',2);
hold on
title('after channel','linewidth',2);
xlabel('Time domain','linewidth',2);
ylabel('Amplitude','linewidth',2);
grid on

f4=figure;
figure(f4);
subplot(2,1,1)
suptitle('consecutive sinc in time domain');
plot(time_domain(1:10*s),fs.*pulse_shaped_signal_in_time_domain(1:10*s),'linewidth',2);
hold on
plot(time_domain(1:10*s),fs.*second_pulse_shaped_signal_in_time_domain(1:10*s),'r','linewidth',2);
hold on
title('before channel','linewidth',2);
xlabel('Time domain','linewidth',2);
ylabel('Amplitude','linewidth',2);
grid on
subplot(2,1,2)
plot(time_domain(1:10*s),fs.*recevied_pulse_shaped_signal_in_time_domain(1:10*s),'linewidth',2);
hold on
plot(time_domain(1:10*s),fs.*recevied_second_pulse_shaped_signal_in_time_domain(1:10*s),'r','linewidth',2);
hold on
title('after channel','linewidth',2);
xlabel('Time domain','linewidth',2);
ylabel('Amplitude','linewidth',2);
grid on

f5=figure;
figure(f5);
subplot(2,1,1)
suptitle('consecutive sinc in frequency domain');
plot(frequency_domain,abs(pulse_shaped_signal_in_frequency_domain),'linewidth',2);
xlim([-5 5]*1e5)
hold on
plot(frequency_domain,abs(second_pulse_shaped_signal_in_frequency_domain),'linewidth',2);
xlim([-5 5]*1e5)
hold on
title('before channel','linewidth',2);
xlabel('Frequency domain','linewidth',2);
ylabel('Amplitude','linewidth',2);
grid on
subplot(2,1,2)
plot(frequency_domain,abs(recevied_pulse_shaped_signal_in_frequency_domain),'linewidth',2);
xlim([-5 5]*1e5)
hold on
plot(frequency_domain,abs(recevied_second_pulse_shaped_signal_in_frequency_domain),'linewidth',2);
xlim([-5 5]*1e5)
hold on
title('after channel','linewidth',2);
xlabel('Frequency domain','linewidth',2);
ylabel('Amplitude','linewidth',2);
grid on
%% 
%%%%%%%%%%%%%% part 2 %%%%%%%%%%%%%%%%%
Energy_per_bit = 1;
Eb_No_dB_vector =(-30:1:0);
N = 1000; 
x_bits  = randi([0 1],1,N);
BPSK = 2*(x_bits)-1; 
H=MultipathChannel(N);
W=((H'*H)\eye(length(H),length(H)))*H';
BER=zeros(1,length(Eb_No_dB_vector));
for i =1:length(Eb_No_dB_vector)
    No= Energy_per_bit/(10^(Eb_No_dB_vector(i)/10));
    Noise=AWGN(BPSK,No);
    Y=H*transpose(BPSK)+transpose(Noise);
    equalized_signal=real(W*Y);
    received_seq=decision(equalized_signal);
    BER(i) = ComputeBER(x_bits,received_seq);
end
display(BER);
f6=figure;
figure(f6);
semilogy(Eb_No_dB_vector,BER,'-ob','linewidth',2)
ylim([10^(-2) 10^0])
legend('BPSK modulation',2)
xlabel('Eb/No','linewidth',2)
ylabel('BER','linewidth',2)
grid on

%%

%%%%%% part 3 %%%%%%
%%%% Repetion channel coding %%% 
p_vect   = 0:0.1:0.5;            % different values of p 
BER_vec  = zeros(size(p_vect));  % the resultant BER
N_bits = 5000; % Total number of bits
r=(1/3); % Code rate
L=(1/r); % Number of samples per symbol (bit)
%-----------------------------------------------------------------
uncoded_bits = randi([0,1],1,N_bits); %generates ranodm integer bits from 0 to 1
%-----------------------------------------------------------------
coded_bits = kron(uncoded_bits,ones(1,L));% Generate samples from bits
%-----------------------------------------------------------------
for p_ind = 1:length(p_vect) 
    %-----------------------------------------------------------------
    % Pass the sample sequence through the channel 
    Received_Coded_bits = zeros(size(coded_bits)); 
    channel_effect = rand(size(Received_Coded_bits))<=p_vect(p_ind);
    Received_Coded_bits = xor(coded_bits,channel_effect);
    %------------------------------------------------------------------
    % Decode bits from received bit sequence
    Estimated_data_bits=zeros(1,length(uncoded_bits));
    v=0;
    for i=1:(length(Received_Coded_bits)/L)
            dh=0;   % dh is the hamming weight 
            for c=1:L
                v=v+1;
                if Received_Coded_bits(v)== 1
                    dh=dh+1; 
                end
            end
            if dh >=(L/2)  
                Estimated_data_bits(i)=1;
            elseif dh <=(L/2)
                Estimated_data_bits(i)=0;
            end
    end
    %-----------------------------------------------------------------
BER_vec(p_ind) = ComputeBER(uncoded_bits,Estimated_data_bits);
end
f7=figure;
figure(f7);
plot(p_vect,BER_vec,'g','linewidth',2); hold on;
xlabel('Values of p','fontsize',10)
ylabel('BER','fontsize',10)
grid on

%%
%%%%%% part 3 %%%%%%
%%%% convolutional code %%% 
%GenerateBits
data = zeros(1,5);
data = round(rand(1,5))

%%%%%%%%%% convelutional encoder%%%%%%%%%
memory=[0 0 0];   % 3 encoder memory elements
encoded_sequence=zeros(1,(length(data))*2);
       
memory(1,3)=memory(1,2);
memory(1,2)=memory(1,1);
memory(1,1)=data(1,1);
temp=xor(memory(1),memory(2));      
X1=xor(temp,memory(3));             %generator polynomial=111 M1 XOR M2 XOR M3
X2=xor(memory(1),memory(3));        %generator polynomial=101 M1 XOR M3
encoded_sequence(1,1)=X1;             
encoded_sequence(1,2)=X2;             
data_len=length(data);
c=3;
for i=2:data_len
         
       memory(1,3)=memory(1,2);
       memory(1,2)=memory(1,1);
       if(i<=data_len)
       memory(1,1)=data(1,i);
       else
       memory(1,1)=0;
       end
              
       temp=xor(memory(1),memory(2));    
       X1=xor(temp,memory(3));
       X2=xor(memory(1),memory(3));
       
       encoded_sequence(1,c)=X1;    %x1 output encoded seq 
       c=c+1;
       encoded_sequence(1,c)=X2;    %x2 output encoded seq 
       c=c+1;
end
output_encoded_sequence=encoded_sequence

%%%%%%%%%%% Viterbi decoder%%%%%%%%%%%

trellis = poly2trellis(3,[7 5]);
tb = 4;
decoded_bits = vitdec(output_encoded_sequence,trellis,tb,'trunc','hard')
