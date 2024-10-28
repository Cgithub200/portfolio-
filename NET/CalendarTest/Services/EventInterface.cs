using Microsoft.AspNetCore.Mvc;
using System.IdentityModel.Tokens.Jwt;
using VertoTest.Models;

namespace VertoTest.Services
{
    public interface EventInterface
    {
        Task<bool> Delete(CalendarEvent content);
        Task<bool> Edit(CalendarEvent existingEvent, EditEventRequest request);
        Task<List<CalendarEvent>> List(DateTime queryDate);
        Task<List<CalendarEvent>> ListRange(DateTime lowerRange, DateTime upperRange);
        Task<bool> Add(CalendarEvent content);
        Task<CalendarEvent?> GetViaId(int id);

        Task<bool> Exists(DateTime lowerRange);

        Task<(CalendarEvent Event, string ErrorMessage)> ValidateAndRetrieveEventByToken(string token, byte[] key);
        string GenerateToken(JwtSecurityTokenHandler tokenHandler, byte[] key, int eventId);

        Task<List<int>> CheckDaysWithEventsInRange(DateTime lowerRange, DateTime upperRange);
    }
}
