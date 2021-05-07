TTS = 0.014
pid = ARGV[0].to_i

loop do
    Process.kill("STOP", pid)
    sleep TTS
    Process.kill("CONT", pid)
    sleep TTS
end
