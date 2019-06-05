# Pulls K2vec and sparsify_codes
include("utils.jl")



function reshape_C(C::Vector{Matrix{Float32}}, m::Integer, h::Integer, d::Integer)
    new_C = Array{Float32, 2}(zeros(m*h, d));
    for i = 1:m
      for j = 1:h
        temp = C[i];
        new_C[(i-1)*h+1:i*h, :] = temp';
      end
    end
    return new_C
  end
  
  function deReshape_C(C::Matrix{Float32}, m::Integer, h::Integer, d::Integer)
    new_C = Vector{Matrix{Float32}}(m);
    for i = 1:m
      for j = 1:h
        temp = C[(i-1)*h+1:i*h, :];
        new_C[i] = temp';
      end
    end
    return new_C
  end
  
  function update_C(X::Matrix{Float32}, B::Matrix{Int16}, h::Integer, C::Vector{Matrix{Float32}})
    lr = 1e-4

    d, n = size(X);
    m, _ = size(B);

    Codes = sparsify_codes(B, h);
    C = reshape_C(C, m, h, d);
    
    C_ = Codes' * (Codes * C - X');
    new_C = C - lr * C_;
  
    new_C = Array{Float32, 2}(new_C);
    C = deReshape_C(new_C, m, h, d);
    return C
  end
  
  
  function SGD_miniBatch(X::Matrix{Float32}, B::Matrix{Int16}, h::Integer, C::Vector{Matrix{Float32}}, lr::AbstractFloat, global_step::Integer, decay_rate::AbstractFloat=0.9, decay_step::Integer=10000, batch_size::Integer=64)
    # init_lr = 0.05
    # decay_rate = 0.9
    # decay_step = 10000
    # batch_size = 64
    # print_every = 256
    # global globalStep = 0

    d, n = size(X);
    m, _ = size(B);

    shuffle_idx = randperm(n)

    X_shuffle = X[:, shuffle_idx];
    B_shuffle = B[:, shuffle_idx];

    Codes = sparsify_codes(B_shuffle, h);
    C = reshape_C(C, m, h, d);
    
    num = Int(floor(n / batch_size))

    for i = 1:num
        bs = Codes[((i-1)*batch_size)+1:(i*batch_size), :];
        xs = X_shuffle[:, ((i-1)*batch_size)+1:(i*batch_size)];
        C_ = bs' * (bs * C - xs');
        C -= lr * C_;
        global_step += 1
        if global_step % decay_step == 0
            print("decay learning rate %.2e -> %.2e\n", lr, lr*decay_rate)
            lr *= decay_rate
        end
    end
  
    new_C = Array{Float32, 2}(C);
    C = deReshape_C(new_C, m, h, d);
    return C, lr, global_step
  end
  
  function Momentum_miniBatch(X::Matrix{Float32}, B::Matrix{Int16}, h::Integer, C::Vector{Matrix{Float32}}, lr::AbstractFloat, global_step::Integer, memory::Matrix{Float32}, momentum::AbstractFloat=0.9, decay_rate::AbstractFloat=0.9, decay_step::Integer=10000, batch_size::Integer=64)
    # init_lr = 0.05
    # decay_rate = 0.9
    # decay_step = 10000
    # batch_size = 64
    # print_every = 256
    # global globalStep = 0

    d, n = size(X);
    m, _ = size(B);

    shuffle_idx = randperm(n)

    X_shuffle = X[:, shuffle_idx];
    B_shuffle = B[:, shuffle_idx];

    Codes = sparsify_codes(B_shuffle, h);
    C = reshape_C(C, m, h, d);
    
    num = Int(floor(n / batch_size))

    for i = 1:num
        bs = Codes[((i-1)*batch_size)+1:(i*batch_size), :];
        xs = X_shuffle[:, ((i-1)*batch_size)+1:(i*batch_size)];
        C_ = bs' * (bs * C - xs');
        memory = momentum * memory + C_
        C -= lr * memory;
        global_step += 1
        if global_step % decay_step == 0
            @printf("decay learning rate %.2e -> %.2e\n", lr, lr*decay_rate)
            lr *= decay_rate
        end
    end
  
    new_C = Array{Float32, 2}(C);
    C = deReshape_C(new_C, m, h, d);
    memory = Array{Float32, 2}(memory);
    return C, lr, global_step, memory
  end

  function Adam_miniBatch(X::Matrix{Float32}, B::Matrix{Int16}, h::Integer, C::Vector{Matrix{Float32}}, lr::AbstractFloat, global_step::Integer, memory::Matrix{Float32}, momentum::AbstractFloat=0.9, decay_rate::AbstractFloat=0.9, decay_step::Integer=10000, batch_size::Integer=64)
    return X
  end
