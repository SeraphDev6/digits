defmodule Mix.Tasks.Train do
  use Mix.Task

  @requirements ["app.start"]

  alias Digits

  def run(_) do
    EXLA.set_as_nx_default([:tpu, :cuda, :rocm, :host])

    {images, labels} = Digits.Model.download()

    images =
      images
      |> Digits.Model.transform_images()
      |> Nx.to_batched(32)

    labels =
      labels
      |> Digits.Model.transform_labels()
      |> Nx.to_batched(32)

    data = Stream.zip(images, labels)

    training_count = floor(0.8 * Enum.count(data))

    {training_data, test_data} = Enum.split(data, training_count)

    model = Digits.Model.new({1, 28, 28})

    Mix.Shell.IO.info("training...")

    state = Digits.Model.train(model, training_data, test_data)

    Digits.Model.save!(model, state)

    :ok
  end
end
