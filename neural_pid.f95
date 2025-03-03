program neural_pid
  implicit none
  integer, parameter :: INPUT_SIZE = 3   ! e, sum_e, delta_e
  integer, parameter :: HIDDEN_SIZE = 4  ! Liczba neuronów w warstwie ukrytej
  integer, parameter :: OUTPUT_SIZE = 1  ! Sygnał sterujący u
  real :: w1(INPUT_SIZE, HIDDEN_SIZE)   ! Wagi wejście-ukryta
  real :: w2(HIDDEN_SIZE, OUTPUT_SIZE)  ! Wagi ukryta-wyjście
  real :: b1(HIDDEN_SIZE)               ! Bias warstwy ukrytej
  real :: b2(OUTPUT_SIZE)               ! Bias warstwy wyjściowej
  real :: x(INPUT_SIZE)                 ! Wejścia: [e, sum_e, delta_e]
  real :: h(HIDDEN_SIZE)                ! Wyjście warstwy ukrytej
  real :: u(OUTPUT_SIZE)                ! Sygnał sterujący (wyjście sieci)
  real :: setpoint = 50.0               ! Wartość zadana (np. temperatura)
  real :: process_value = 20.0          ! Aktualna wartość procesu
  real :: error, sum_error, delta_error ! Błąd, suma błędów, zmiana błędu
  real :: last_error = 0.0              ! Poprzedni błąd
  real :: learning_rate = 0.01          ! Współczynnik uczenia
  integer :: i, epoch, timestep
  real :: process_gain = 0.1            ! Wzmocnienie procesu (symulacja)

  ! Inicjalizacja wag i biasów losowymi wartościami
  call random_seed()
  call random_number(w1)
  call random_number(w2)
  call random_number(b1)
  call random_number(b2)
  w1 = 0.2 * (2.0 * w1 - 1.0)  ! Skalowanie do [-0.2, 0.2]
  w2 = 0.2 * (2.0 * w2 - 1.0)
  b1 = 0.2 * (2.0 * b1 - 1.0)
  b2 = 0.2 * (2.0 * b2 - 1.0)

  ! Symulacja procesu i uczenie sieci
  sum_error = 0.0
  do timestep = 1, 1000
    ! Obliczenie błędu, sumy błędów i zmiany błędu
    error = setpoint - process_value
    sum_error = sum_error + error
    delta_error = error - last_error

    ! Przygotowanie wejść do sieci
    x(1) = error
    x(2) = sum_error
    x(3) = delta_error

    ! Propagacja w przód
    h = matmul(x, w1) + b1
    h = tanh(h)  ! Funkcja aktywacji tanh zamiast sigmoid dla symetrii
    u = matmul(h, w2) + b2

    ! Symulacja procesu (np. nagrzewanie)
    process_value = process_value + process_gain * u(1)
    if (process_value < 0.0) process_value = 0.0  ! Ograniczenie fizyczne

    ! Uczenie sieci (propagacja wsteczna)
    call backpropagation(x, h, u, error, w1, w2, b1, b2, learning_rate)

    ! Wyświetlanie co 50 kroków
    if (mod(timestep, 50) == 0) then
      print *, "Krok:", timestep, "PV:", process_value, "U:", u(1), "E:", error
    end if

    ! Zapamiętanie poprzedniego błędu
    last_error = error
  end do

contains
  ! Funkcja aktywacji tanh
  function tanh(x) result(res)
    real, intent(in) :: x(:)
    real :: res(size(x))
    res = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
  end function tanh

  ! Pochodna tanh
  function tanh_derivative(x) result(res)
    real, intent(in) :: x(:)
    real :: res(size(x))
    res = 1.0 - ((exp(x) - exp(-x)) / (exp(x) + exp(-x)))**2
  end function tanh_derivative

  ! Propagacja wsteczna
  subroutine backpropagation(x, h, u, target_error, w1, w2, b1, b2, lr)
    real, intent(in) :: x(:), h(:), u(:), target_error, lr
    real, intent(inout) :: w1(:,:), w2(:,:), b1(:), b2(:)
    real :: delta_out(OUTPUT_SIZE), delta_hidden(HIDDEN_SIZE)
    real :: error(OUTPUT_SIZE)

    ! Błąd na wyjściu (proporcjonalny do błędu procesu)
    error = target_error  ! Używamy błędu procesu jako sygnału uczącego
    delta_out = error * 1.0  ! Wyjście jest liniowe, brak aktywacji na końcu

    ! Aktualizacja wag i biasów warstwy wyjściowej
    w2 = w2 - lr * spread(h, dim=2, ncopies=OUTPUT_SIZE) * spread(delta_out, dim=1, ncopies=HIDDEN_SIZE)
    b2 = b2 - lr * delta_out

    ! Błąd warstwy ukrytej
    delta_hidden = matmul(w2, delta_out) * tanh_derivative(h)

    ! Aktualizacja wag i biasów warstwy ukrytej
    w1 = w1 - lr * spread(x, dim=2, ncopies=HIDDEN_SIZE) * spread(delta_hidden, dim=1, ncopies=INPUT_SIZE)
    b1 = b1 - lr * delta_hidden
  end subroutine backpropagation
end program neural_pid
